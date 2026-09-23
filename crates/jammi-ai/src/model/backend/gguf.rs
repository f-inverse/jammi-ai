//! GGUF/k-quant loading and residency-estimation helpers shared by the
//! resolver (header-only residency estimation), the candle backend's GGUF
//! load path, and the fine-tune QLoRA backbone load (`fine_tune::worker`) —
//! ONE definition of which tensor names are "matmul-site" for a supported
//! architecture, which GGML dtypes this workspace can represent, and which
//! `config.json` field names encode an architecture's geometry
//! ([`normalize_model_config`]) and transformer layer count
//! ([`gguf_num_layers`]), so none of these consumers can silently drift on
//! any of the three questions. Reading `num_hidden_layers`/`num_layers` off
//! the RAW config is right for BERT-family/ModernBERT but silently wrong for
//! DistilBERT, whose config.json declares only the DistilBERT-native
//! `n_layers`: an estimator that sees zero layers empties `matmul_site` and
//! costs every k-quant weight as dense — ~7x on the matmul-site WEIGHT BYTES
//! alone, ~1.9x on this module's own fixture's TOTAL residency once
//! biases/embeddings/the dequantize transient are folded in (see
//! `gguf_residency_estimate_is_meaningfully_smaller_than_its_own_all_dense_fallback_per_architecture`,
//! whose fixture measures 85888 correctly-classified vs. 166912 all-dense
//! bytes) — and a fine-tune load hard-refuses outright.
//!
//! # The three supported architectures
//!
//! GGUF/quantized serving is threaded only through the three text towers
//! `jammi_encoders::FrozenWeightLookup` is wired into: BERT-family
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

/// The three text-tower architectures GGUF loading is threaded
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

/// The byte width [`estimate_gguf_residency`] costs every densified
/// (non-matmul-site) GGUF tensor at, unconditionally, by design (see that
/// function's doc): the WIDEST byte width any
/// [`ComputePrecision`](jammi_numerics::ComputePrecision) variant can take.
/// `F32`'s 4 bytes is that value ([`compute_precision_byte_size`]'s own
/// match: `F32 => 4`, `F16`/`BF16 => 2`) — this helper COMPUTES it via a
/// single exhaustive match with one compile-forced arm per variant, and NO
/// separate candidate list: a `[F32, F16, BF16]` array literal is a
/// hand-maintained list a new variant could be added without ever touching.
/// Each arm below recurses into the NEXT variant's own arm, so a new
/// `ComputePrecision` variant fails to compile inside this match — not
/// merely inside [`compute_precision_byte_size`]'s own match one level up —
/// until this function is deliberately revisited to decide where the new
/// variant folds in.
pub(crate) fn widest_compute_precision_byte_size() -> usize {
    use jammi_numerics::ComputePrecision::{BF16, F16, F32};
    fn widest_from(precision: jammi_numerics::ComputePrecision) -> usize {
        let own_bytes = compute_precision_byte_size(precision);
        match precision {
            F32 => own_bytes.max(widest_from(F16)),
            F16 => own_bytes.max(widest_from(BF16)),
            BF16 => own_bytes,
        }
    }
    widest_from(F32)
}

/// Normalize DistilBERT config field names to the BERT-standard set every
/// consumer expects. DistilBERT's `config.json` uses `dim`/`n_heads`/
/// `n_layers`/`hidden_dim`/`dropout`/`attention_dropout`/`activation` where a
/// BERT-family config uses `hidden_size`/`num_attention_heads`/
/// `num_hidden_layers`/`intermediate_size`/`hidden_dropout_prob`/
/// `attention_probs_dropout_prob`/`hidden_act`, and omits `type_vocab_size`/
/// `layer_norm_eps` entirely (defaulted here to DistilBERT's own
/// documented values). A private helper: every caller goes through
/// [`normalize_model_config`], which is the one that knows WHEN
/// normalization applies (only `model_type == "distilbert"`).
fn normalize_distilbert_config(config: &serde_json::Value) -> serde_json::Value {
    let mut normalized = config.clone();
    if let Some(obj) = normalized.as_object_mut() {
        // Field renames: DistilBERT → BERT
        let mappings: &[(&str, &str)] = &[
            ("dim", "hidden_size"),
            ("n_heads", "num_attention_heads"),
            ("n_layers", "num_hidden_layers"),
            ("hidden_dim", "intermediate_size"),
            ("dropout", "hidden_dropout_prob"),
            ("attention_dropout", "attention_probs_dropout_prob"),
        ];
        for &(src, dst) in mappings {
            if let Some(val) = obj.get(src).cloned() {
                obj.entry(dst).or_insert(val);
            }
        }
        // activation → hidden_act (string value)
        if let Some(val) = obj.get("activation").cloned() {
            obj.entry("hidden_act").or_insert(val);
        }
        // Defaults for fields DistilBERT doesn't have but BertConfig requires
        obj.entry("type_vocab_size")
            .or_insert(serde_json::Value::from(2));
        obj.entry("layer_norm_eps")
            .or_insert(serde_json::json!(1e-12));
    }
    normalized
}

/// THE single config-normalization authority (module doc): every consumer
/// that reads architecture geometry off a `config.json` `Value` — the
/// ordinary (non-GGUF) DistilBERT encoder-config deserialize in
/// `CandleBackend::load`, [`gguf_num_layers`] (and through it
/// [`estimate_gguf_residency`] and both the fine-tune and inference GGUF
/// backbone loads) — MUST route the raw config through this function first.
/// A DistilBERT `config.json` declares its layer count (and every other
/// BERT-standard field) under DistilBERT-native names ONLY
/// ([`normalize_distilbert_config`]'s own doc); reading a raw,
/// un-normalized DistilBERT config for `num_hidden_layers` silently sees
/// nothing, which makes DistilBERT GGUF QLoRA unreachable and empties the
/// residency estimator's matmul-site set. A no-op clone for every other
/// `model_type`.
pub(crate) fn normalize_model_config(
    model_type: &str,
    config: &serde_json::Value,
) -> serde_json::Value {
    if model_type == "distilbert" {
        normalize_distilbert_config(config)
    } else {
        config.clone()
    }
}

/// THE single site every GGUF consumer reads a checkpoint's transformer
/// layer count through (module doc): normalizes `config` via
/// [`normalize_model_config`] FIRST, then reads `num_hidden_layers`
/// (falling back to the pre-normalization `num_layers` name some
/// checkpoints still carry directly at the top level). `None` when neither
/// field is present after normalization — every call site turns that into
/// its own typed refusal naming the checkpoint/model, never a silent
/// default (a `0`-layer fallback here would empty the matmul-site set and
/// cost every k-quant weight as dense — ~7x on the matmul-site weight
/// bytes alone, ~1.9x on this module's own fixture's total residency (see
/// `gguf_residency_estimate_is_meaningfully_smaller_than_its_own_all_dense_fallback_per_architecture`)
/// — and refuse models that actually fit).
pub(crate) fn gguf_num_layers(model_type: &str, config: &serde_json::Value) -> Option<usize> {
    let normalized = normalize_model_config(model_type, config);
    normalized
        .get("num_hidden_layers")
        .or_else(|| normalized.get("num_layers"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
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
/// source). A parse failure here is a typed RESOLVER refusal — never a silent fallback to the raw
/// file size, which would under-report residency for a file whose header lies about its own tensor
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

/// RESOLVE-TIME residency estimation: parses ONLY `path`'s GGUF header
/// (never tensor data — see [`read_gguf_header`]) and returns a
/// CONSERVATIVE (>= true residency) byte figure, one category per
/// [`load_gguf_backbone`] branch (derived from that function's actual code):
///
/// - a matmul-site WEIGHT tensor stored at a genuinely block-quantized
///   dtype stays resident as a `QTensor` (`load_gguf_backbone`'s
///   `SiteWeight::Quantized` arm — NEVER dequantized) — costed at its own
///   storage size (`elem_count / block_size * type_size`) plus the CUDA
///   row-padding candle's quantized CUDA loader always applies
///   ([`MATRIX_ROW_PADDING`], this module's own doc) — applied
///   unconditionally (not gated on the actual target device), because this
///   estimate must stay conservative regardless of which device the model
///   eventually loads on;
/// - a matmul-site BIAS is UNCONDITIONALLY dequantized at load
///   (`load_gguf_backbone`'s bias arm calls `.dequantize(device)` on every
///   matmul-site bias regardless of its own stored dtype — never a
///   `SiteWeight::Quantized` case), so it is costed on the SAME dense path
///   as every other non-matmul-site-weight tensor, below — NEVER the
///   compressed-`QTensor` path, even when the bias itself happens to be
///   stored at a genuinely block-quantized dtype (a pathological but
///   representable GGUF checkpoint);
/// - every OTHER tensor (a non-matmul-site tensor of any stored dtype, a
///   matmul-site WEIGHT stored densely, or ANY matmul-site BIAS) is
///   dequantized at load — resident as a plain `Tensor`, `elem_count *
///   `[`widest_compute_precision_byte_size`]`()` — see the "# Densified
///   tensors are costed at the widest representable dtype, by design"
///   section below for why this is a FIXED width, never a caller-supplied
///   one;
/// - PLUS the single largest per-tensor DEQUANTIZE transient
///   (`stored_bytes + 4·N + widest_compute_precision_byte_size()·N` —
///   candle's own `QTensor::dequantize` always produces an `F32` tensor
///   FIRST, `4·N`, before any narrowing cast to the target dtype, so both
///   buffers can be momentarily resident together) across every
///   dequantized tensor (including every matmul-site bias) — the
///   worst-case peak overlap during load, added ONCE (not per-tensor).
///
/// A malformed header (an element count that is not a multiple of its
/// dtype's own block size — the same domain check `TensorInfo::read` itself
/// applies when it actually reads bytes) or an unrepresented GGML dtype
/// (neither a `WeightQuantization` k-quant format nor `F32`/`F16`/`BF16`) is
/// a typed refusal here too, at RESOLVE time — never silently estimated
/// past a shape this workspace cannot actually load.
///
/// # Densified tensors are costed at the widest representable dtype, by design
///
/// The dense-cost width is not a parameter. This function runs at resolve
/// time, before `ResolvedModel.adapter_path` (a fine-tuned model's adapter,
/// if any) is even consulted, and a saved fine-tune adapter carries its OWN
/// persisted `backbone_dtype` (a training-time choice —
/// `CandleBackend::load`'s `encoder_backbone_dtype` prefers it over
/// `compute_precision` whenever an adapter is present, see that call site's
/// own comment) which can be WIDER than the resolve-time `compute_precision`
/// (e.g. an F32-trained adapter served under an F16 `compute_precision`
/// config). Costing every densified tensor at
/// [`widest_compute_precision_byte_size`]`()` keeps the ">= true residency"
/// invariant through that adapter-backbone-dtype window; a caller-supplied
/// width derived from the resolve-time `compute_precision` could only
/// narrow the figure below what an adapter load actually produces. The
/// price is a deliberately conservative figure for a narrow-precision
/// config:
///
/// - on this module's own one-layer BERT fixture (the fixture
///   `estimate_gguf_residency_costs_every_densified_tensor_at_f32_width_unconditionally`
///   builds: 6 matmul-site Q4_0 weights + 6 F32 biases + one `[128,32]`
///   F32 embedding), an F16 `compute_precision` config's admission figure
///   — MEASURED at F16 width (2 bytes/element) — would be 58560 bytes;
///   MEASURED at F32 width (the figure this function reports) it is
///   75712 bytes, a +29% rise
///   (`(75712 - 58560) / 58560 ≈ 0.293`) — both figures reproduced by that
///   test's `estimator_exact_residency_at` closure at `width` = 2 and 4
///   respectively (an EXACT replica of this function's own arithmetic,
///   including `MATRIX_ROW_PADDING`; the module's separate
///   `true_residency_contribution` parity helper deliberately omits that
///   padding term to stay a conservative lower bound, so it is never the
///   producer of these two exact figures);
/// - the dense (non-matmul-site) TERM specifically scales linearly with
///   dtype width (`elem_count * width`), but the max-transient term's own
///   `4 * elem_count` component is width-independent, so the admission
///   figure's F32-vs-F16 ratio is NOT the full 2x the per-element width
///   itself moves by: on this fixture (the `[128,32]` F32 embedding is the
///   entire non-matmul-site dense mass, not merely dominant among several), the MEASURED
///   ratio is `75712 / 58560 ≈ 1.29` (the same +29% rise cited above) —
///   the full 2x is only approached in the opposite regime, where the
///   single largest densified tensor is negligible relative to the total
///   densified byte count across every non-matmul-site tensor, which this
///   fixture does not exercise.
///
/// # The non-GGUF safetensors residency path
///
/// `CandleBackend::load` loads a safetensors checkpoint at
/// `compute_dtype`/`effective_precision` regardless of what dtype it was
/// SAVED at (candle.rs's `VarBuilder::from_mmaped_safetensors` call), so a
/// plain file-size sum is dtype-blind: an F16-on-disk checkpoint served
/// under the `F32` default is resident at roughly 2x its file-byte sum. (The
/// `F32` default that `candle.rs`'s
/// `per_model_precision.unwrap_or(device_config.compute_precision)` falls
/// back to is produced not by `ComputePrecision`'s own `#[default]` arm but
/// by `GpuConfig`'s manual `impl Default` hardcoding
/// `compute_precision: jammi_numerics::ComputePrecision::F32` — reached via
/// `DeviceConfig::from_config` ← `config.gpu.compute_precision`, whose
/// missing-field fallback `GpuConfig`'s container-level
/// `#[serde(default, deny_unknown_fields)]` routes to `GpuConfig::default()`,
/// never to `ComputePrecision::default()` — a second hand-written `F32`
/// default that can drift from the first.)
///
/// `ModelResolver`'s three resolve paths and
/// `CandleBackend::estimate_memory` therefore route a safetensors checkpoint
/// through [`super::safetensors_residency::estimate_safetensors_residency`],
/// which parses ONLY the safetensors header (never tensor data — the same
/// "header, not data" shape this function uses) and costs every tensor at
/// `elem_count * max(on_disk_dtype_bytes, `[`widest_compute_precision_byte_size`]`())`
/// — the SAME widest-width clamp this function applies to every densified
/// GGUF tensor, for the same reason (conservative under every reachable
/// effective precision, including a persisted adapter's own
/// `backbone_dtype`). `jammi_ai::model::backend::ort`'s own residency
/// estimator still separately applies a 1.3x multiplier over its file-size
/// sum (`OrtBackend::estimate_memory`) — a constant this module's own
/// reasoning neither derives nor explains (the ORT backend is unavailable
/// in this build regardless — `OrtBackend::load` always returns a typed
/// refusal).
pub(crate) fn estimate_gguf_residency(
    path: &Path,
    model_config: &serde_json::Value,
    model_id: &str,
) -> Result<usize> {
    // Costed at the WIDEST representable dtype, unconditionally — see this
    // function's own "Densified tensors are costed at the widest
    // representable dtype, by design" doc section for why this is not a
    // caller-supplied parameter.
    let target_dtype_bytes = widest_compute_precision_byte_size();
    let content = read_gguf_header(path, model_id)?;

    // The SAME shared reader `CandleBackend::load` and the fine-tune worker's
    // GGUF arm dispatch on (`crate::model::arch::config_model_type`): the
    // absent-`model_type` default is owned once, in `model::arch`, alongside
    // the FAMILY answer `EncoderFamily::from_config` gives for the identical
    // config — never re-typed as a local `unwrap_or("bert")` here, which could
    // drift from it.
    let model_type = crate::model::arch::config_model_type(model_config);
    // ONLY matmul-site WEIGHT names — never `.bias` — per
    // [`load_gguf_backbone`]'s own branches (this function's doc, category
    // parity): a matmul-site WEIGHT stored at a genuinely block-quantized
    // dtype loads as a resident `QTensor` and is costed compressed below;
    // a matmul-site BIAS is UNCONDITIONALLY dequantized at load
    // (`load_gguf_backbone`'s bias arm calls `.dequantize(device)`
    // regardless of the bias tensor's own stored dtype), so it must be
    // costed on the SAME dense/transient path as every other tensor —
    // folding `.bias` into this set would cost a genuinely quantized bias as
    // a resident compressed `QTensor` with no dequantize transient, UNDER
    // the true residency the loader actually produces, breaking this
    // function's own ">= true residency" invariant. A dense (`F32`)-stored
    // bias — the common case — makes the distinction a no-op, so only a
    // quantized-bias fixture exercises it.
    let matmul_site_weights: HashSet<String> = match GgufArchitecture::from_model_type(model_type) {
        Some(arch) => {
            // Routes through the SAME normalization + layer-count authority
            // ([`gguf_num_layers`]) the fine-tune and inference GGUF
            // backbone loads use — a raw, un-normalized DistilBERT config
            // (whose only layer-count field is the DistilBERT-native
            // `n_layers`) must refuse here exactly the way an actual load
            // would, never silently fall back to `0` layers (which would
            // empty `matmul_site_weights` and cost every k-quant weight as
            // dense — ~7x on the matmul-site weight bytes alone, ~1.9x on
            // this module's own fixture's total residency (see
            // `gguf_residency_estimate_is_meaningfully_smaller_than_its_own_all_dense_fallback_per_architecture`)).
            let num_layers = gguf_num_layers(model_type, model_config).ok_or_else(|| {
                refusal(
                    model_id,
                    format!(
                        "GGUF residency estimation requires num_hidden_layers (or num_layers) \
                         in config.json for a {arch:?} backbone"
                    ),
                )
            })?;
            matmul_site_names(arch, &content.tensor_infos, num_layers)
                .into_iter()
                .map(|(name, _has_bias)| format!("{name}.weight"))
                .collect()
        }
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
        if is_kquant && matmul_site_weights.contains(name) {
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
/// whatever else the checkpoint carries). The checkpoint's quantization
/// format is a header fact, read by [`gguf_modal_quantization`] when the
/// model is described, never here.
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
/// DATA (unlike [`estimate_gguf_residency`] and
/// [`gguf_modal_quantization`], this reads and — for non-matmul-site
/// tensors — dequantizes every tensor's bytes). The checkpoint's tensors
/// are checked by [`checkpoint_matmul_sites`] first, the same check the
/// description applied.
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

    let site_specs = checkpoint_matmul_sites(&content, arch, num_layers, weights_path, model_id)?;

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
        let weight = if weight_quantization_from_ggml(ggml_dtype).is_some() {
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
    })
}

/// The matmul sites a `{arch}` backbone with `num_layers` layers requires
/// of `content`'s checkpoint, as `(prefix, has_bias)`, once every tensor in
/// the file has cleared the dtype pre-flight. Shared by the description
/// ([`gguf_modal_quantization`]) and the load ([`load_gguf_backbone`]), so
/// the two refuse the same checkpoint the same way.
///
/// Refuses (typed, naming tensor + dtype) when any tensor carries a GGML
/// dtype this workspace does not represent (neither a k-quant
/// [`WeightQuantization`] format nor `F32`/`F16`/`BF16`) — the densified
/// path must be able to dequantize whatever it finds, and a refusal up
/// front is clearer than a deep candle error mid-densify — and refuses
/// (typed, LISTING every missing name) when any required matmul-site
/// tensor is absent.
fn checkpoint_matmul_sites(
    content: &gguf_file::Content,
    arch: GgufArchitecture,
    num_layers: usize,
    weights_path: &Path,
    model_id: &str,
) -> Result<Vec<(String, bool)>> {
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
    let missing: Vec<String> = site_specs
        .iter()
        .flat_map(|(prefix, has_bias)| {
            let weight = format!("{prefix}.weight");
            let bias = has_bias.then(|| format!("{prefix}.bias"));
            std::iter::once(weight).chain(bias)
        })
        .filter(|name| !content.tensor_infos.contains_key(name))
        .collect();
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
    Ok(site_specs)
}

/// The MODAL quantized dtype among the matmul-site tensors of the GGUF
/// checkpoint at `weights_path`, read from its header alone (never tensor
/// data — see [`read_gguf_header`]): the value `ModelIdentity.quantization`
/// reports. Ties break by [`WeightQuantization`]'s own `Ord` (that type's
/// module doc — the GGUF wire ID). A file whose matmul-site tensors are
/// ALL stored densely (F32/F16/BF16, no genuine k-quant tensor at all — a
/// pathological, self-defeating "GGUF" checkpoint) reports `None` rather
/// than fabricating a quantized format that is never used.
pub(crate) fn gguf_modal_quantization(
    weights_path: &Path,
    arch: GgufArchitecture,
    num_layers: usize,
    model_id: &str,
) -> Result<Option<WeightQuantization>> {
    let content = read_gguf_header(weights_path, model_id)?;
    let site_specs = checkpoint_matmul_sites(&content, arch, num_layers, weights_path, model_id)?;
    let mut counts: HashMap<WeightQuantization, usize> = HashMap::new();
    for (prefix, _) in &site_specs {
        let info = &content.tensor_infos[&format!("{prefix}.weight")];
        if let Some(quantization) = weight_quantization_from_ggml(info.ggml_dtype) {
            *counts.entry(quantization).or_insert(0) += 1;
        }
    }
    Ok(counts
        .into_iter()
        .max_by(|(a, a_n), (b, b_n)| a_n.cmp(b_n).then_with(|| a.cmp(b)))
        .map(|(quantization, _)| quantization))
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
        use jammi_numerics::ComputePrecision::{BF16, F16, F32};
        // Exhaustive match (no wildcard `_` arm): this
        // test fails to COMPILE, not merely fails to pass, the moment a
        // new `ComputePrecision` variant is added — forcing that variant's
        // expected byte width to be pinned here (and
        // `widest_compute_precision_byte_size` revisited, see the
        // cross-check below) before either function can be trusted again.
        let mut expected_widest = 0usize;
        for precision in [F32, F16, BF16] {
            let expected = match precision {
                F32 => 4,
                F16 => 2,
                BF16 => 2,
            };
            assert_eq!(compute_precision_byte_size(precision), expected);
            expected_widest = expected_widest.max(expected);
        }
        // Cross-checks `widest_compute_precision_byte_size` against the
        // widest of THIS test's own exhaustively-pinned expected widths
        // (`expected_widest`, folded in the loop above) — NOT a second
        // `[F32, F16, BF16]` candidate array. That function derives every
        // variant's width via its own exhaustive recursive match rather than
        // a hand-maintained list (see its doc); this assertion must not
        // re-introduce such a list on the test side.
        assert_eq!(widest_compute_precision_byte_size(), expected_widest);
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

    // ─────────────────────────────────────────────────────────────────
    // `normalize_model_config` / `gguf_num_layers` (RAW vs. NORMALIZED
    // config): the single authority every consumer routes through.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn gguf_num_layers_reads_a_raw_distilbert_configs_native_n_layers_field() {
        let raw = serde_json::json!({ "model_type": "distilbert", "n_layers": 4 });
        assert_eq!(gguf_num_layers("distilbert", &raw), Some(4));
    }

    #[test]
    fn gguf_num_layers_reads_num_hidden_layers_for_bert_and_modernbert() {
        let bert = serde_json::json!({ "model_type": "bert", "num_hidden_layers": 3 });
        assert_eq!(gguf_num_layers("bert", &bert), Some(3));
        let modernbert = serde_json::json!({ "model_type": "modernbert", "num_hidden_layers": 5 });
        assert_eq!(gguf_num_layers("modernbert", &modernbert), Some(5));
    }

    #[test]
    fn gguf_num_layers_is_none_when_a_distilbert_config_carries_neither_layer_field() {
        // A raw DistilBERT config read WITHOUT normalization: only the
        // BERT-standard names are absent, `n_layers` itself is what the
        // fixture omits here, so this pins the "genuinely missing" case
        // distinct from "present under the native name".
        let raw = serde_json::json!({ "model_type": "distilbert", "dim": 32 });
        assert_eq!(gguf_num_layers("distilbert", &raw), None);
    }

    #[test]
    fn normalize_model_config_is_a_no_op_clone_for_non_distilbert_architectures() {
        let bert = serde_json::json!({ "model_type": "bert", "num_hidden_layers": 2 });
        assert_eq!(normalize_model_config("bert", &bert), bert);
        let modernbert = serde_json::json!({ "model_type": "modernbert", "num_hidden_layers": 2 });
        assert_eq!(
            normalize_model_config("modernbert", &modernbert),
            modernbert
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // `estimate_gguf_residency` oracles: a real GGUF header, parsed end to
    // end, for each of two axes — (a) a raw config's layer count, (b) the
    // per-category costing parity with `load_gguf_backbone`.
    // ─────────────────────────────────────────────────────────────────

    /// FNV-1a-seeded deterministic small-magnitude values.
    fn est_fixture_tensor(name: &str, dims: &[usize], device: &Device) -> Tensor {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let seed = h as f64;
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| (((seed % 97.0) + 1.0) * (i as f64) * 0.037 + seed * 1e-6).sin() as f32 * 0.1)
            .collect();
        Tensor::from_vec(v, dims, device).unwrap()
    }

    /// The RAW (native-field-name) config.json a real checkpoint of `arch`
    /// ships, with `layers` transformer layers — DistilBERT spelled with
    /// its own `n_layers`, on purpose (the shape an un-normalized read
    /// silently mis-reads).
    fn est_raw_config(arch: GgufArchitecture, layers: usize) -> serde_json::Value {
        match arch {
            GgufArchitecture::Bert => {
                serde_json::json!({ "model_type": "bert", "num_hidden_layers": layers })
            }
            GgufArchitecture::DistilBert => {
                serde_json::json!({ "model_type": "distilbert", "n_layers": layers })
            }
            GgufArchitecture::ModernBert => {
                serde_json::json!({ "model_type": "modernbert", "num_hidden_layers": layers })
            }
        }
    }

    /// Write a `model.gguf` for `arch` with `layers` transformer layers:
    /// every matmul-site WEIGHT quantized at `quant`, every matmul-site
    /// BIAS and one embedding-shaped "everything else" tensor dense
    /// (`F32`)-stored. Uses a FIXED tensor shape for every matmul-site
    /// tensor (the estimator's arithmetic doesn't depend on which logical
    /// role a tensor plays, only its element count and dtype) —
    /// deliberately decoupled from `jammi_encoders`' exact per-layer
    /// shapes, since this function tests `estimate_gguf_residency` in
    /// isolation, never a real load.
    fn write_est_fixture(
        dir: &Path,
        arch: GgufArchitecture,
        layers: usize,
        quant: GgmlDType,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        std::fs::create_dir_all(dir).unwrap();
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut matmul_weight_names: Vec<String> = Vec::new();
        for (prefix, has_bias) in matmul_site_names(arch, &HashMap::new(), layers) {
            let w = format!("{prefix}.weight");
            tensors.insert(w.clone(), est_fixture_tensor(&w, &[64, 32], device));
            matmul_weight_names.push(w);
            if has_bias {
                let b = format!("{prefix}.bias");
                tensors.insert(b.clone(), est_fixture_tensor(&b, &[64], device));
            }
        }
        tensors.insert(
            "embeddings.word_embeddings.weight".to_string(),
            est_fixture_tensor("embeddings.word_embeddings.weight", &[128, 32], device),
        );

        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort(); // deterministic write order
        let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
        for name in &names {
            let t = &tensors[*name];
            let dtype = if matmul_weight_names.contains(name) {
                quant
            } else {
                GgmlDType::F32
            };
            qtensors.push(((*name).clone(), QTensor::quantize(t, dtype).unwrap()));
        }
        let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        gguf_file::write(&mut writer, &[], &refs).unwrap();
        tensors
    }

    /// Axis (a): reading `num_hidden_layers`/`num_layers` directly off the
    /// RAW config makes a raw DistilBERT config (whose only layer-count
    /// field is `n_layers`) produce `num_layers = 0`, emptying `matmul_site`
    /// and costing every k-quant weight as if it were dense — IDENTICAL to
    /// what an unsupported architecture's all-dense fallback produces. This
    /// test isolates the mechanism: the ONLY difference between the two
    /// `estimate_gguf_residency` calls below is `model_type` — same GGUF
    /// file, same `target_dtype_bytes`. A fixed (non-empty) matmul-site
    /// set MUST cost strictly less than the all-dense fallback (Q4_0
    /// compresses far below F32); an accidentally-empty matmul-site set
    /// would make the two numbers IDENTICAL.
    #[test]
    fn distilbert_gguf_residency_estimate_costs_matmul_site_weights_compressed_not_dense() {
        let device = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("distilbert_model");
        write_est_fixture(
            &dir,
            GgufArchitecture::DistilBert,
            2,
            GgmlDType::Q4_0,
            &device,
        );
        let path = dir.join("model.gguf");

        let raw_distilbert_config = est_raw_config(GgufArchitecture::DistilBert, 2);
        let distilbert_estimate =
            estimate_gguf_residency(&path, &raw_distilbert_config, "distilbert-model").unwrap();

        // Forces the EXACT all-dense fallback path a zero-layer read takes
        // for DistilBERT: an unrecognized `model_type`
        // makes `GgufArchitecture::from_model_type` return `None`, so
        // `matmul_site_weights` is empty and every tensor — including the
        // Q4_0-quantized matmul-site weights — is costed dense.
        let unsupported_config = serde_json::json!({ "model_type": "some_unrecognized_arch" });
        let all_dense_estimate =
            estimate_gguf_residency(&path, &unsupported_config, "distilbert-model").unwrap();

        assert!(
            distilbert_estimate < all_dense_estimate,
            "a correctly-classified DistilBERT matmul-site set must cost strictly less than the \
             all-dense fallback (compressed Q4_0 weights vs. dense F32) — got \
             distilbert={distilbert_estimate} all_dense={all_dense_estimate}; equal values mean \
             matmul_site_weights was silently empty (the all-dense fallback)"
        );
        // MEASURED: Q4_0 stores ~4.5 bits/weight vs. F32's 32 —
        // roughly a 7x compression on the matmul-site weight bytes alone,
        // so a wide (not tuned-to-merely-pass), but still meaningful,
        // floor: the fixed cost must drop by at least a third.
        assert!(
            (distilbert_estimate as f64) < (all_dense_estimate as f64) * 0.7,
            "expected a meaningfully smaller compressed estimate, got \
             distilbert={distilbert_estimate} all_dense={all_dense_estimate}"
        );
    }

    /// On this fixture (2 layers, `[64,32]` Q4_0 matmul-site weights,
    /// `[64]` F32 matmul-site biases, one `[128,32]` F32 embedding,
    /// `target_dtype_bytes = 4`), MEASURED (not assumed) by this
    /// function's own arithmetic below:
    ///
    /// - BERT/DistilBERT correctly-classified estimate: 85888 bytes
    ///   (identical for both — same tensor geometry, only the tensor-name
    ///   strings differ between the two architectures' fixtures).
    /// - ModernBERT correctly-classified estimate: 77056 bytes (fewer,
    ///   bias-free sites).
    /// - The all-dense fallback (every matmul-site weight, including the
    ///   Q4_0-quantized ones, costed dense — forced via an unrecognized
    ///   `model_type`, which is EXACTLY what an empty `matmul_site_weights`
    ///   set produces): 166912 bytes for BERT/DistilBert's geometry.
    ///
    /// A mis-classifying estimator makes DistilBERT's estimate equal ITS
    /// OWN all-dense fallback (166912), not BERT's correctly-classified
    /// figure. A peer-ratio bound such as `(0.3..3.0).contains(&(db /
    /// peer))` is vacuous against that: `166912 / 85888 ≈ 1.943` lies
    /// inside it. This test instead asserts, per architecture,
    /// `correctly_classified < 0.7 * own_all_dense_fallback`, which an
    /// empty `matmul_site_weights` fails by construction (it makes
    /// `correctly_classified == all_dense`, never `< 0.7 *` itself).
    #[test]
    fn gguf_residency_estimate_is_meaningfully_smaller_than_its_own_all_dense_fallback_per_architecture(
    ) {
        let device = Device::Cpu;
        let layers = 2;
        let unsupported_config = serde_json::json!({ "model_type": "some_unrecognized_arch" });
        for arch in [
            GgufArchitecture::Bert,
            GgufArchitecture::DistilBert,
            GgufArchitecture::ModernBert,
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("model");
            write_est_fixture(&dir, arch, layers, GgmlDType::Q4_0, &device);
            let path = dir.join("model.gguf");
            let config = est_raw_config(arch, layers);
            let correctly_classified = estimate_gguf_residency(&path, &config, "model").unwrap();
            // Forces the all-dense fallback: an unrecognized `model_type`
            // makes `matmul_site_weights` empty regardless of architecture
            // — the SAME state a `0`-layer / mis-normalized-config bug
            // would silently produce for a SUPPORTED architecture (module
            // doc; the sibling
            // `distilbert_gguf_residency_estimate_costs_matmul_site_weights_compressed_not_dense`
            // test isolates that exact mechanism for DistilBERT).
            let all_dense_fallback =
                estimate_gguf_residency(&path, &unsupported_config, "model").unwrap();
            assert!(
                (correctly_classified as f64) < (all_dense_fallback as f64) * 0.7,
                "{arch:?}: correctly-classified estimate {correctly_classified} must be \
                 meaningfully smaller than its own all-dense fallback {all_dense_fallback} \
                 (bound: < 0.7x); a value at or near the fallback means \
                 matmul_site_weights was silently empty (the all-dense fallback)"
            );
        }

        // Authority-derived, non-vacuous positive control (a `db > 0.0`
        // check is unconditionally true — embeddings alone make it
        // positive regardless of classification): calls the SAME
        // `matmul_site_names` authority `estimate_gguf_residency` calls
        // internally (module doc), with the SAME `layers` value the loop
        // above built its DistilBERT fixture with. The `&HashMap::new()` below (rather than a
        // live GGUF file's parsed `content.tensor_infos`) does NOT weaken
        // this control for DistilBERT specifically — `matmul_site_names`'s
        // `DistilBert` arm (unlike its `Bert` arm, which consults
        // `tensor_infos` via `bert_prefix` to detect a `bert.`-wrapper
        // tensor) never reads `tensor_infos` at all, so an empty map and a
        // real file's `tensor_infos` produce IDENTICAL DistilBERT output —
        // this control genuinely constrains the same computation the
        // estimator performs for this architecture, not merely a
        // superficially similar one. The matmul-site NAME authority must
        // derive exactly 12 DistilBERT sites (6 per layer × 2 layers) for
        // this fixture — the exact count `estimate_gguf_residency` needs
        // non-empty to avoid the all-dense fallback above.
        let distilbert_sites =
            matmul_site_names(GgufArchitecture::DistilBert, &HashMap::new(), layers);
        assert_eq!(
            distilbert_sites.len(),
            12,
            "authority-derived DistilBERT matmul-site count must be exactly 12 (6 sites/layer \
             × 2 layers) for this fixture's geometry"
        );
    }

    /// Domain-validity refusal: a SUPPORTED architecture whose
    /// config carries neither layer-count field is a typed refusal at
    /// estimate time, never a silent `0`-layer / empty-matmul-site
    /// fallback that would under-cost every k-quant weight.
    #[test]
    fn gguf_residency_estimate_refuses_a_supported_architecture_missing_the_layer_count_field() {
        let device = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        write_est_fixture(
            &dir,
            GgufArchitecture::DistilBert,
            1,
            GgmlDType::Q4_0,
            &device,
        );
        let config = serde_json::json!({ "model_type": "distilbert" }); // no n_layers at all

        let err = estimate_gguf_residency(&dir.join("model.gguf"), &config, "model")
            .expect_err("a supported architecture missing its layer-count field must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("num_hidden_layers") || msg.contains("num_layers"),
            "refusal must name the missing field, got: {msg}"
        );
    }

    /// A GGUF `config.json` LACKING `model_type` entirely: the shared reader
    /// (`crate::model::arch::config_model_type`, the same one
    /// `CandleBackend::load` and the fine-tune worker's GGUF arm read through)
    /// answers with `crate::model::arch::UNDECLARED_MODEL_TYPE_FAMILY`'s id,
    /// `"bert"` — a SUPPORTED architecture, so this does NOT hit the
    /// `None`-architecture all-dense fallback branch. A config missing BOTH `model_type` and the
    /// BERT-standard layer-count field must still refuse via the SAME typed error the sibling test
    /// above pins for an EXPLICIT `model_type: "distilbert"`, never silently
    /// succeed on a `0`-layer / empty-`matmul_site_weights` estimate.
    #[test]
    fn gguf_residency_estimate_refuses_a_config_lacking_model_type_entirely() {
        let device = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        write_est_fixture(&dir, GgufArchitecture::Bert, 1, GgmlDType::Q4_0, &device);
        let config = serde_json::json!({}); // no model_type, no layer-count field at all

        let err = estimate_gguf_residency(&dir.join("model.gguf"), &config, "model").expect_err(
            "a config missing model_type (defaulting to bert) AND its layer-count field \
                 must refuse",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("num_hidden_layers") || msg.contains("num_layers"),
            "refusal must name the missing field, got: {msg}"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Estimator/loader per-category parity (axis (b)): a TABLE-DRIVEN test
    // over every tensor category `load_gguf_backbone` actually branches on,
    // INCLUDING a genuinely quantized bias — the only state in which the
    // two can drift apart.
    // ─────────────────────────────────────────────────────────────────

    /// One category `load_gguf_backbone` branches on: a tensor of `elems`
    /// elements, stored at `stored_dtype`, and whether it is a matmul-site
    /// WEIGHT or a matmul-site BIAS (mutually exclusive; both false for a
    /// non-matmul-site tensor).
    struct Category {
        name: String,
        elems: usize,
        stored_dtype: GgmlDType,
        is_matmul_site_weight: bool,
        is_matmul_site_bias: bool,
    }

    /// Independent per-tensor true-residency-contribution derivation, from
    /// [`load_gguf_backbone`]'s own branches (never from
    /// [`estimate_gguf_residency`]'s code — this function is never called
    /// by that one, so this test cannot pass by tautologically re-deriving
    /// the function under test): `(total_bytes, transient_candidate)` for
    /// ONE tensor at `target_dtype_bytes`.
    fn true_residency_contribution(c: &Category, target_dtype_bytes: usize) -> (u128, u128) {
        debug_assert!(
            !(c.is_matmul_site_weight && c.is_matmul_site_bias),
            "'{}' cannot be both a matmul-site weight and a matmul-site bias",
            c.name
        );
        let is_kquant = weight_quantization_from_ggml(c.stored_dtype).is_some();
        let elems = c.elems as u128;
        if c.is_matmul_site_weight && is_kquant {
            // `SiteWeight::Quantized` — resident `QTensor`, its own
            // block-quantized storage size, no dequantize transient.
            let block = c.stored_dtype.block_size() as u128;
            let type_size = c.stored_dtype.type_size() as u128;
            (elems / block * type_size, 0)
        } else {
            // Every other category — `SiteWeight::Dense`, a matmul-site
            // BIAS (ALWAYS dequantized regardless of `is_kquant` — the
            // exact case this test targets), or a non-matmul-site tensor
            // — is resident as a plain dense `Tensor`, with a dequantize
            // transient when the stored bytes differ from the dense ones.
            let block = c.stored_dtype.block_size() as u128;
            let type_size = c.stored_dtype.type_size() as u128;
            let stored_bytes = elems / block * type_size;
            let dense_bytes = elems * target_dtype_bytes as u128;
            let transient = stored_bytes + elems * 4 + dense_bytes;
            (dense_bytes, transient)
        }
    }

    #[test]
    fn estimate_gguf_residency_is_at_least_true_residency_for_every_loader_category() {
        let device = Device::Cpu;
        // Derived from the SAME authority `estimate_gguf_residency`
        // costs every densified tensor at internally (never a bare literal
        // `4` re-typed here — that would silently drift the moment the
        // authority's own value changes).
        let target_dtype_bytes = widest_compute_precision_byte_size();
        let layers = 1;

        let mut categories = vec![
            Category {
                name: "encoder.layer.0.attention.self.query.weight".to_string(),
                elems: 64 * 32,
                stored_dtype: GgmlDType::Q4_0,
                is_matmul_site_weight: true,
                is_matmul_site_bias: false,
            },
            Category {
                // A GENUINELY quantized matmul-site bias: `load_gguf_backbone` dequantizes it
                // unconditionally (never a resident `QTensor`), so its true residency is
                // `elems * target_dtype_bytes`, not a compressed size. Sized
                // large enough (4096 elements, a realistic hidden-size-ish
                // width) that the fixed `MATRIX_ROW_PADDING` overhead does
                // NOT dominate — for a tiny bias the padding alone can
                // exceed the dense cost, which would make a (wrongly)
                // compressed cost LARGER than the correct dense one and hide
                // the drift from a `>=` check entirely.
                name: "encoder.layer.0.attention.self.query.bias".to_string(),
                elems: 4096,
                stored_dtype: GgmlDType::Q4_0,
                is_matmul_site_weight: false,
                is_matmul_site_bias: true,
            },
            Category {
                // A matmul-site weight stored densely (F32) — the loader's
                // `SiteWeight::Dense` arm.
                name: "encoder.layer.0.attention.self.key.weight".to_string(),
                elems: 64 * 32,
                stored_dtype: GgmlDType::F32,
                is_matmul_site_weight: true,
                is_matmul_site_bias: false,
            },
            Category {
                // A non-matmul-site tensor (embeddings) — always densified.
                name: "embeddings.word_embeddings.weight".to_string(),
                elems: 128 * 32,
                stored_dtype: GgmlDType::F32,
                is_matmul_site_weight: false,
                is_matmul_site_bias: false,
            },
            Category {
                // A QUANTIZED NON-matmul-site tensor (e.g. k-quant token-type
                // embeddings — real GGUF checkpoints DO quantize embedding
                // tables, not just matmul-site linear weights).
                // `load_gguf_backbone`'s densify loop (module doc) calls `.dequantize(device)`
                // UNCONDITIONALLY on every tensor whose name isn't in
                // `site_names` — the SAME path a dense-stored non-site tensor
                // takes, never `SiteWeight::Quantized` (that arm is reachable
                // ONLY for a matmul-site WEIGHT). `estimate_gguf_residency`'s
                // own branch condition (`is_kquant && matmul_site_weights.
                // contains(name)`) agrees — a k-quant tensor NOT in
                // `matmul_site_weights` falls through to the same dense/
                // transient `else` arm — and this row pins that agreement.
                name: "embeddings.token_type_embeddings.weight".to_string(),
                elems: 64 * 32,
                stored_dtype: GgmlDType::Q4_0,
                is_matmul_site_weight: false,
                is_matmul_site_bias: false,
            },
        ];

        // Every OTHER matmul-site tensor this fixture's `layers=1` BERT
        // shape requires but the explicit table above doesn't name
        // (`matmul_site_names` names six sites/layer; the table names
        // two) — filled in as dense (F32) categories, so
        // `load_gguf_backbone`'s own missing-tensor refusal never fires
        // AND every one of these participates in the SAME independent
        // true-residency derivation below (never silently excluded from
        // the accounting the way a hand-picked subset would be).
        let required = matmul_site_names(GgufArchitecture::Bert, &HashMap::new(), layers);
        let named: std::collections::HashSet<&str> =
            categories.iter().map(|c| c.name.as_str()).collect();
        let mut extra_names: Vec<String> = Vec::new();
        for (prefix, has_bias) in &required {
            let w = format!("{prefix}.weight");
            if !named.contains(w.as_str()) {
                extra_names.push(w);
            }
            if *has_bias {
                let b = format!("{prefix}.bias");
                if !named.contains(b.as_str()) {
                    extra_names.push(b);
                }
            }
        }
        for name in extra_names {
            let is_weight = name.ends_with(".weight");
            categories.push(Category {
                name,
                elems: if is_weight { 64 * 32 } else { 64 },
                stored_dtype: GgmlDType::F32,
                is_matmul_site_weight: is_weight,
                is_matmul_site_bias: !is_weight,
            });
        }

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for c in &categories {
            tensors.insert(
                c.name.clone(),
                est_fixture_tensor(&c.name, &[c.elems], &device),
            );
        }

        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        std::fs::create_dir_all(&dir).unwrap();
        let category_dtype: HashMap<&str, GgmlDType> = categories
            .iter()
            .map(|c| (c.name.as_str(), c.stored_dtype))
            .collect();
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
        for name in &names {
            let dtype = category_dtype.get(name.as_str()).copied().unwrap();
            let t = &tensors[*name];
            qtensors.push(((*name).clone(), QTensor::quantize(t, dtype).unwrap()));
        }
        let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        gguf_file::write(&mut writer, &[], &refs).unwrap();

        let config = est_raw_config(GgufArchitecture::Bert, layers);
        let estimate = estimate_gguf_residency(&dir.join("model.gguf"), &config, "model").unwrap();

        // Independent per-category true-residency derivation (table-driven
        // — a fixture that later adds a category updates the table, not
        // ad hoc arithmetic), including the SAME "largest transient added
        // once" accounting `estimate_gguf_residency`'s own doc describes,
        // computed here from the category table alone.
        let mut true_total: u128 = 0;
        let mut true_max_transient: u128 = 0;
        for c in &categories {
            let (bytes, transient) = true_residency_contribution(c, target_dtype_bytes);
            true_total += bytes;
            if transient > true_max_transient {
                true_max_transient = transient;
            }
        }
        let true_residency = true_total + true_max_transient;

        assert!(
            (estimate as u128) >= true_residency,
            "estimate {estimate} must be >= the per-category derived true residency \
             {true_residency} (loader-category table: {:?})",
            categories
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // F32-width costing is UNCONDITIONAL, not caller-controlled:
    // `estimate_gguf_residency` takes no width parameter and costs every
    // densified tensor at [`widest_compute_precision_byte_size`]`()` (see
    // that function's doc and `estimate_gguf_residency`'s own "Densified
    // tensors are costed at the widest representable dtype, by design"
    // section). There is no width to call with, so this test pins the
    // exact byte figures the doc discloses.
    // ─────────────────────────────────────────────────────────────────

    /// Pins the interface property `estimate_gguf_residency`'s own doc
    /// discloses: on this fixture (one-layer BERT — 6 matmul-site Q4_0
    /// weights, 6 F32 biases, one `[128,32]` F32 embedding), the function
    /// reports EXACTLY the F32-width figure (75712 bytes), never the
    /// narrower F16-width figure (58560 bytes) an F16 dense-cost width
    /// would produce. Both figures are MEASURED here, via
    /// `estimator_exact_residency_at` — an independent REPLICA of
    /// `estimate_gguf_residency`'s own per-tensor arithmetic (including the
    /// `MATRIX_ROW_PADDING` term the `true_residency_contribution` parity
    /// helper deliberately omits, see that closure's own doc), built fresh
    /// from this fixture's own category list, so this cannot pass by
    /// tautologically re-deriving the function under test — at `width` = 4
    /// and 2 respectively, not assumed or merely transcribed from the doc.
    #[test]
    fn estimate_gguf_residency_costs_every_densified_tensor_at_f32_width_unconditionally() {
        let device = Device::Cpu;
        let layers = 1;
        let arch = GgufArchitecture::Bert;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("model");
        write_est_fixture(&dir, arch, layers, GgmlDType::Q4_0, &device);
        let path = dir.join("model.gguf");
        let config = est_raw_config(arch, layers);

        let estimate = estimate_gguf_residency(&path, &config, "model").unwrap();

        // Independent per-category true-residency derivation — mirrors
        // this fixture's own geometry (`write_est_fixture`'s doc: six
        // BERT sites/layer, one `[128,32]` F32 embedding).
        let required = matmul_site_names(arch, &HashMap::new(), layers);
        let mut categories: Vec<Category> = Vec::new();
        for (prefix, has_bias) in &required {
            categories.push(Category {
                name: format!("{prefix}.weight"),
                elems: 64 * 32,
                stored_dtype: GgmlDType::Q4_0,
                is_matmul_site_weight: true,
                is_matmul_site_bias: false,
            });
            if *has_bias {
                categories.push(Category {
                    name: format!("{prefix}.bias"),
                    elems: 64,
                    stored_dtype: GgmlDType::F32,
                    is_matmul_site_weight: false,
                    is_matmul_site_bias: true,
                });
            }
        }
        categories.push(Category {
            name: "embeddings.word_embeddings.weight".to_string(),
            elems: 128 * 32,
            stored_dtype: GgmlDType::F32,
            is_matmul_site_weight: false,
            is_matmul_site_bias: false,
        });

        // A faithful REPLICA of `estimate_gguf_residency`'s own internal
        // arithmetic — NOT the `true_residency_contribution` LOWER-BOUND
        // helper the `>=`-style parity tests above use (that helper
        // deliberately omits `MATRIX_ROW_PADDING` for a matmul-site
        // quantized weight, since it only ever needs to be a conservative
        // floor, never an exact match). This closure exists to reproduce
        // the estimator's EXACT output at a given dense-cost width, so
        // this test can pin the precise doc-disclosed figures at both
        // widths.
        let estimator_exact_residency_at = |width: usize| -> u128 {
            let mut total: u128 = 0;
            let mut max_transient: u128 = 0;
            for c in &categories {
                let block = c.stored_dtype.block_size() as u128;
                let type_size = c.stored_dtype.type_size() as u128;
                let elems = c.elems as u128;
                let is_kquant = weight_quantization_from_ggml(c.stored_dtype).is_some();
                if c.is_matmul_site_weight && is_kquant {
                    let padding = MATRIX_ROW_PADDING as u128 * type_size / block;
                    total += elems / block * type_size + padding;
                } else {
                    let stored_bytes = elems / block * type_size;
                    let dense_bytes = elems * width as u128;
                    total += dense_bytes;
                    let transient = stored_bytes + elems * 4 + dense_bytes;
                    if transient > max_transient {
                        max_transient = transient;
                    }
                }
            }
            total + max_transient
        };

        let residency_at_f32 = estimator_exact_residency_at(4);
        let residency_at_f16 = estimator_exact_residency_at(2);

        // MEASURED, not transcribed: the SAME two figures
        // `estimate_gguf_residency`'s own doc discloses in its "Densified
        // tensors are costed at the widest representable dtype, by
        // design" section.
        assert_eq!(residency_at_f16, 58560);
        assert_eq!(residency_at_f32, 75712);

        assert_eq!(
            estimate as u128, residency_at_f32,
            "estimate_gguf_residency must report EXACTLY the F32-width figure \
             ({residency_at_f32}) unconditionally — never the narrower \
             F16-width figure ({residency_at_f16}); F32-width costing is this \
             function's own behavior, not something a caller can narrow"
        );
    }

    /// The sharpest axis-(b) oracle: two fixtures, IDENTICAL except for the
    /// stored dtype of ONE matmul-site bias tensor (genuinely block-
    /// quantized `Q4_0` vs. dense `F32`). `load_gguf_backbone` dequantizes
    /// a matmul-site bias UNCONDITIONALLY regardless of its own stored
    /// dtype (module doc) — its true residency contribution is therefore
    /// IDENTICAL either way, so a correct estimator must report the SAME
    /// total for both fixtures. An estimator that folds `.bias` into the
    /// matmul-site set sends a Q4_0-stored bias down the compressed
    /// `QTensor` branch (strictly smaller than the dense branch) while an
    /// F32-stored one takes the dense branch, so the two estimates DIFFER.
    /// This isolates the mechanism (change only the claimed cause, confirm
    /// the number does not move) far more precisely than any absolute `>=`
    /// bound can.
    #[test]
    fn quantized_and_dense_matmul_site_bias_cost_identically_since_the_loader_always_densifies_it()
    {
        let device = Device::Cpu;
        let layers = 1;
        let config = est_raw_config(GgufArchitecture::Bert, layers);
        let required = matmul_site_names(GgufArchitecture::Bert, &HashMap::new(), layers);
        let bias_override_name = format!("{}.bias", required[0].0);

        let build_and_estimate = |bias_dtype: GgmlDType| -> usize {
            let mut tensors: HashMap<String, Tensor> = HashMap::new();
            for (prefix, has_bias) in &required {
                let w = format!("{prefix}.weight");
                tensors.insert(w.clone(), est_fixture_tensor(&w, &[64, 32], &device));
                if *has_bias {
                    let b = format!("{prefix}.bias");
                    tensors.insert(b.clone(), est_fixture_tensor(&b, &[64], &device));
                }
            }
            tensors.insert(
                "embeddings.word_embeddings.weight".to_string(),
                est_fixture_tensor("embeddings.word_embeddings.weight", &[128, 32], &device),
            );

            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("model");
            std::fs::create_dir_all(&dir).unwrap();
            let mut names: Vec<&String> = tensors.keys().collect();
            names.sort();
            let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
            for name in &names {
                let dtype = if **name == bias_override_name {
                    bias_dtype
                } else if name.ends_with(".weight") && *name != "embeddings.word_embeddings.weight"
                {
                    GgmlDType::Q4_0
                } else {
                    GgmlDType::F32
                };
                let t = &tensors[*name];
                qtensors.push(((*name).clone(), QTensor::quantize(t, dtype).unwrap()));
            }
            let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
            let mut writer = std::io::BufWriter::new(file);
            let refs: Vec<(&str, &QTensor)> =
                qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
            gguf_file::write(&mut writer, &[], &refs).unwrap();

            estimate_gguf_residency(&dir.join("model.gguf"), &config, "model").unwrap()
        };

        let quantized_bias_estimate = build_and_estimate(GgmlDType::Q4_0);
        let dense_bias_estimate = build_and_estimate(GgmlDType::F32);

        assert_eq!(
            quantized_bias_estimate, dense_bias_estimate,
            "a matmul-site bias's stored dtype must not change the estimate — \
             `load_gguf_backbone` dequantizes every matmul-site bias unconditionally, so its \
             residency contribution is the SAME dense cost regardless of whether the bytes on \
             disk happen to be block-quantized; a difference here means the bias was costed as \
             a resident compressed QTensor (estimator/loader category drift)"
        );
    }
}
