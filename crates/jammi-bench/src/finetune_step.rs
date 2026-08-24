//! The encoder fine-tune step tier: how long one LoRA training step takes, and
//! what it costs in memory.
//!
//! ## Why this tier exists next to `train-scale`
//!
//! [`crate::train_scale`] measures the GradCache in-batch-negative path over a
//! projection head on synthetic embeddings — it proves a *bounded activation
//! footprint* and a CPU pairs/s rate, and it never touches a real encoder. Every
//! optimization that matters for encoder fine-tuning (attention masking, the
//! backbone dtype the base GEMM actually runs at, how the dropout mask is
//! produced, the softmax path) lives inside the encoder forward and backward,
//! which that tier does not execute.
//!
//! This tier executes exactly that: three encoder forwards (anchor, positive,
//! negative — all three live on the tape simultaneously, as the trainer keeps
//! them), a cosine-margin triplet loss, one backward into the LoRA tensors, and
//! one AdamW step. It is the unit a PyTorch + PEFT reference loop measures, so
//! the two are comparable step-for-step.
//!
//! ## What is gated and what is recorded
//!
//! Nothing here is gated. A step time is a property of `code x device x box`,
//! and this tier is meant to run on a rented GPU whose model is not pinned —
//! exactly the condition under which the previous absolute GPU floor
//! false-failed and was removed. It **records**, tagged with the device that
//! produced it, so two runs on the *same* box (a parent commit and a change) can
//! be compared as a ratio. That within-run A/B is the only comparison a
//! heterogeneous fleet supports.
//!
//! ## Honesty about what is measured
//!
//! The optimizer, the LoRA layers, and the encoder are the engine's own. The
//! triplet loss is re-implemented here because the trainer's is crate-private —
//! the same re-implementation licence [`crate::train_scale`] takes for
//! `mnrl_loss`, and the same arithmetic: `mean(relu(margin - cos(a,p) +
//! cos(a,n)))` over L2-normalized pooled embeddings.
//!
//! Token ids are synthetic and uniform over the vocabulary. That is deliberate:
//! this tier measures *cost*, not learning, and a fixed synthetic batch removes
//! tokenizer and data-loading variance from a number meant to isolate the
//! compute path. It is therefore not a quality measurement and must never be
//! quoted as one.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarMap};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::report::{FinetuneStepTier, Measurement};

/// Poll total device memory in use, in bytes, via `nvidia-smi`.
///
/// Whole-device, not per-process: on a dedicated pod this session is the only
/// consumer, and the tier subtracts a baseline read after the model is resident,
/// so the reported figure is activation and workspace growth. On a shared GPU it would over-report, so the field
/// is documented as device-total-minus-baseline rather than as a process
/// measurement.
fn device_memory_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|mib| mib * 1024 * 1024)
}

/// Sample device memory on a background thread for the duration of the measured
/// steps, so the reported peak is the real high-water mark rather than whatever
/// happened to be allocated when the last step ended.
struct VramSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VramSampler {
    fn start() -> Option<Self> {
        device_memory_used_bytes()?;
        let peak = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (Arc::clone(&peak), Arc::clone(&stop));
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(used) = device_memory_used_bytes() {
                    p.fetch_max(used, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        });
        Some(Self {
            peak,
            stop,
            handle: Some(handle),
        })
    }

    fn finish(mut self, baseline: u64) -> Measurement {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let peak = self.peak.load(Ordering::Relaxed);
        Measurement::measured(peak.saturating_sub(baseline) as f64, "bytes")
    }
}

/// Cosine-margin triplet loss over three L2-normalized `[batch, hidden]` blocks.
///
/// `mean(relu(margin - cos(a, p) + cos(a, n)))`. Rows are already unit-norm
/// (every encoder `forward` ends in `pool_and_normalize`), so the cosine is a
/// row-wise dot product.
fn triplet_loss(a: &Tensor, p: &Tensor, n: &Tensor, margin: f64) -> candle_core::Result<Tensor> {
    let pos = (a * p)?.sum(candle_core::D::Minus1)?;
    let neg = (a * n)?.sum(candle_core::D::Minus1)?;
    let raw = ((neg - pos)? + margin)?;
    raw.relu()?.mean_all()
}

/// Which of the encoders this tier can build (`modernbert` | `bert` |
/// `distilbert`) a checkpoint directory holds.
///
/// This is NOT the trainer's full family list: the trainer's own dispatch
/// (`jammi_ai::fine_tune::worker::build_encoder_adapters`) has a `_ =>
/// Bert` catch-all, and `model/backend/candle.rs` enumerates `bert`,
/// `roberta`, `camembert`, and `xlm-roberta` as BERT-family — every one of
/// those siblings loads through the same `Bert` builder this tier drives
/// (see [`Self::BERT_FAMILY_SIBLINGS`]). A checkpoint whose `config.json`
/// names one of those siblings is driven by passing `--model-type bert`
/// explicitly ([`Self::detect`] accepts this — a sibling's own `model_type`
/// is not one of this tier's three, so it is not a "genuine mislabel" the
/// way naming `distilbert` while `--model-type bert` is passed would be);
/// the file's own string survives in the report as `config_model_type` so
/// the row stays attributable to what the checkpoint actually claims to be.
///
/// This is loading, not parity: the `Bert` builder probes only the `bert.`
/// tensor-key prefix (or an unprefixed root layout) and knows nothing of
/// RoBERTa's padding_idx-offset position ids — driving a RoBERTa
/// checkpoint through `--model-type bert` measures the step cost of the
/// weights it loaded, not a claim that HF's RoBERTa forward and this one
/// agree.
///
/// Detected from `config.json`'s `model_type` field the same way the trainer
/// reads it (`crates/jammi-ai/src/fine_tune/worker.rs`,
/// `model_config.get("model_type").and_then(|v| v.as_str())`) — except this
/// tier refuses an absent or unrecognized value instead of the trainer's
/// `.unwrap_or("bert")` fallback. A benchmark that silently mis-classified
/// the model under test would corrupt the per-model profiling this tier
/// exists for (issue #356, rule 12: profile first, per model), so "which
/// model did I just measure" must never be a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    ModernBert,
    Bert,
    DistilBert,
}

/// What [`ModelType::detect`] resolved, plus the two raw inputs it
/// resolved FROM — kept as two separate fields rather than one collapsed
/// `Option<String>` because a single field cannot distinguish "declared and
/// agreed" (`config_model_type = None` under the old shape, because the
/// file's declaration already equalled the resolved model) from "declared
/// nothing at all, the operator asserted `--model-type`" (also `None`
/// under the old shape, for the opposite reason — there was nothing to
/// diverge from). A durable JSON record must be able to reconstruct which
/// of those happened; two fields that each mean exactly one thing can, one
/// collapsed field cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedModel {
    pub model_type: ModelType,
    /// `config.json`'s own `model_type` string, EXACTLY as read from the
    /// file. `None` ONLY when the field is literally absent — including
    /// when the file's declaration already equals the resolved
    /// [`Self::model_type`] (agreement is a fact the record preserves, not
    /// a state worth collapsing to `None`).
    pub config_model_type: Option<String>,
    /// The `--model-type` override EXACTLY as passed on the command line.
    /// `None` ONLY when the flag was never supplied — including when a
    /// supplied override agreed with `config.json`'s own declaration (a
    /// no-op override is still a fact about how this row was produced).
    pub model_type_override: Option<String>,
}

impl ModelType {
    pub const ALL: [&'static str; 3] = ["modernbert", "bert", "distilbert"];

    /// BERT-family siblings this tier does not autodetect (their
    /// `model_type` is not literally one of [`Self::ALL`]) but accepts via
    /// an explicit `--model-type bert` override, because they load through
    /// the same `Bert` builder as `"bert"` itself. Sourced from
    /// `crates/jammi-ai/src/model/backend/candle.rs`'s
    /// `"bert" | "roberta" | "camembert" | "xlm-roberta"` match arm and
    /// `crates/jammi-ai/src/fine_tune/worker.rs`'s `_ => Bert` catch-all —
    /// the tier's own trainer/inference dispatch, not a list invented here.
    pub const BERT_FAMILY_SIBLINGS: [&'static str; 3] = ["roberta", "camembert", "xlm-roberta"];

    fn parse(raw: &str, source: ModelTypeSource) -> Result<Self, ModelTypeError> {
        match raw {
            "modernbert" => Ok(Self::ModernBert),
            "bert" => Ok(Self::Bert),
            "distilbert" => Ok(Self::DistilBert),
            other => Err(ModelTypeError::Unknown {
                value: other.to_string(),
                source,
            }),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ModernBert => "modernbert",
            Self::Bert => "bert",
            Self::DistilBert => "distilbert",
        }
    }

    /// Detect from a parsed `config.json`. An explicit `--model-type`
    /// override wins whenever the file's own `model_type` is not one of
    /// this tier's own three ([`Self::ALL`]): `model_type` CAN be absent
    /// from a real checkpoint (`jammi_encoders::BertConfig`'s own
    /// `model_type` field is `#[serde(default)]` `Option<String>` — a raw
    /// `BertModel` checkpoint need not carry it — and the trainer's own
    /// dispatch treats an absent field as `"bert"` rather than refusing to
    /// build), and it can also be a BERT-family sibling
    /// ([`Self::BERT_FAMILY_SIBLINGS`]) or any other value this tier does
    /// not itself build — in every one of those cases the override is the
    /// only way to say which builder to use, so it is accepted, and the
    /// file's own string is preserved in the returned
    /// [`DetectedModel::config_model_type`] rather than silently dropped.
    ///
    /// Only when `config.json` DOES declare one of this tier's own three
    /// types and the override names a *different* one is that a genuine
    /// mislabel, not a gap the override exists to fill: silently trusting
    /// the override there would let the tier profile the wrong builder
    /// while reporting the override's label, corrupting the per-model
    /// classification this tier exists for — refuse with a typed error
    /// naming both instead.
    pub fn detect(
        config_json: &serde_json::Value,
        override_type: Option<&str>,
    ) -> Result<DetectedModel, ModelTypeError> {
        let file_type = config_json.get("model_type").and_then(|v| v.as_str());
        // Parse the override to a `ModelType` FIRST, before it is compared
        // against anything `config.json` declares. An invalid `--model-type`
        // value (not one of `Self::ALL`) must be refused as an invalid FLAG
        // — `ModelTypeError::Unknown { source: OverrideFlag, .. }` — even
        // when `config.json` happens to declare one of this tier's own
        // three types; comparing the raw, unvalidated string first (the
        // previous shape) let an invalid override reach the
        // `OverrideContradicts` guard purely because it differed from a
        // valid file declaration, surfacing "config.json declares X but
        // --model-type was Y" for a Y that was never a real model type to
        // begin with — the wrong surface, pointing at the wrong flag.
        let parsed_override = override_type
            .map(|explicit| Self::parse(explicit, ModelTypeSource::OverrideFlag))
            .transpose()?;
        let model_type = match (parsed_override, file_type) {
            (Some(explicit), Some(file_raw))
                if Self::ALL.contains(&file_raw) && file_raw != explicit.as_str() =>
            {
                return Err(ModelTypeError::OverrideContradicts {
                    file_type: file_raw.to_string(),
                    override_type: explicit.as_str().to_string(),
                });
            }
            (Some(explicit), _) => explicit,
            (None, Some(raw)) => Self::parse(raw, ModelTypeSource::ConfigJson)?,
            (None, None) => return Err(ModelTypeError::Absent),
        };
        Ok(DetectedModel {
            model_type,
            // Exactly as read from the file — `None` only when the field is
            // literally absent (see `DetectedModel::config_model_type`'s
            // doc; agreement with `model_type` is preserved, not collapsed).
            config_model_type: file_type.map(str::to_string),
            // Exactly as passed on the command line — `None` only when the
            // flag was never supplied.
            model_type_override: override_type.map(str::to_string),
        })
    }

    /// The LoRA target-module selector vocabulary this model's encoder
    /// actually wires up — the real per-layer module names each encoder's
    /// `LoraSite::build`/`LoraSlot::build_in` call passes as its FIRST
    /// argument, the string `jammi_lora::should_apply_lora` matches against
    /// (NOT the second argument, the LoRA adapter subpath used only to
    /// address the trainable A/B tensors — those are a different vocabulary
    /// and matching against it selects nothing, since `should_apply_lora`
    /// never sees it).
    ///
    /// `bert.rs:536-548`: `attention.self.query`, `attention.self.key`,
    /// `attention.self.value`, `attention.output.dense`,
    /// `intermediate.dense`, `output.dense`. `distilbert.rs`'s six call
    /// sites pass short, mutually-exclusive names directly (`q_lin`,
    /// `k_lin`, `v_lin`, `out_lin`, `lin1`, `lin2`) so no suffix expansion
    /// applies there. `modernbert.rs:1178-1215`: `Wqkv`, `Wo`, `Wi`,
    /// `mlp.Wo`.
    ///
    /// `should_apply_lora` matches a target string against the END of the
    /// module name (`module_name == t || module_name.ends_with(t)`), so two
    /// selectors here are ambiguous BY THAT SEMANTICS, not by a bug in this
    /// vocabulary: ModernBERT's `"Wo"` also matches `mlp.Wo` (`"mlp.Wo"`
    /// ends with `"Wo"`), and BERT's `"output.dense"` also matches
    /// `attention.output.dense` (`"attention.output.dense"` ends with
    /// `"output.dense"`) — both are the trainer's own selector semantics,
    /// not something this tier tries to disambiguate. A caller that wants
    /// exactly one of an ambiguous pair must pass the longer, unambiguous
    /// form (`attention.output.dense` selects only the attention-output
    /// site; there is no suffix-only selector that reaches the FFN
    /// `output.dense` without also reaching `attention.output.dense`).
    pub fn lora_target_vocabulary(self) -> &'static [&'static str] {
        match self {
            Self::ModernBert => &["Wqkv", "Wo", "Wi"],
            Self::Bert => &[
                "query",
                "key",
                "value",
                "attention.output.dense",
                "intermediate.dense",
                "output.dense",
            ],
            Self::DistilBert => &["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which surface an unrecognized `model_type` string was actually read
/// from — `config.json`'s own field, or the `--model-type` flag. Carried by
/// [`ModelTypeError::Unknown`] so its refusal message names the surface
/// that actually needs fixing rather than always addressing `config.json`
/// (the message a `--model-type` typo used to get, pointing the operator at
/// the wrong file).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTypeSource {
    ConfigJson,
    OverrideFlag,
}

/// Why [`ModelType::detect`] could not resolve a model family, or why the
/// tensor-prefix probe inside [`run`] refused before constructing an
/// encoder. Typed (not a bare string) so a caller matching on the failure
/// never has to string-match a message.
#[derive(Debug)]
pub enum ModelTypeError {
    /// A `model_type` string was not one of [`ModelType::ALL`] — the
    /// `source` field says whether it came from `config.json`'s own field
    /// (with no `--model-type` override supplied to resolve it — a known
    /// BERT-family sibling falls in here too, absent an override) or from
    /// an invalid `--model-type` value itself.
    Unknown {
        value: String,
        source: ModelTypeSource,
    },
    /// `config.json` has no `model_type` field and no `--model-type`
    /// override was supplied.
    Absent,
    /// `config.json` names one of this tier's own three types
    /// ([`ModelType::ALL`]) and `--model-type` names a *different*, itself
    /// VALID one — a genuine mislabel, not a gap. The override exists to
    /// resolve a checkpoint whose own declaration is absent, a BERT-family
    /// sibling ([`ModelType::BERT_FAMILY_SIBLINGS`]), or anything else this
    /// tier does not itself build; it is not a licence to relabel a
    /// checkpoint that already declares itself one of the three, since that
    /// would drive the wrong builder while the report still carries the
    /// override's label. An INVALID override never reaches this variant —
    /// [`ModelType::detect`] parses (and can therefore refuse) the override
    /// before it is ever compared against `config.json`.
    OverrideContradicts {
        file_type: String,
        override_type: String,
    },
    /// The checkpoint's tensor keys carry a prefix the `Bert` builder's own
    /// two-layout probe does not know how to load — it loads only a
    /// `"bert."`-wrapped `BertForX` checkpoint or an unprefixed root
    /// `BertModel` checkpoint. A BERT-family sibling saved through its own
    /// save_pretrained (e.g. `RobertaForX`) emits keys under its own
    /// class name, a third layout this tier cannot drive through
    /// `--model-type bert`; widening `jammi-encoders`' `bert.rs` prefix
    /// probe to accept sibling prefixes is a known follow-up, not something
    /// this tier can work around.
    UnsupportedTensorPrefix { prefix: String },
}

impl std::fmt::Display for ModelTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown {
                value,
                source: ModelTypeSource::ConfigJson,
            } => write!(
                f,
                "unknown model_type {value:?} in config.json; expected one of {:?}, or pass \
                 --model-type explicitly. Known BERT-family siblings {:?} are accepted this way \
                 (e.g. --model-type bert for a roberta checkpoint) — they load through the Bert \
                 builder (prefix `bert.` or root layout); RoBERTa-family position-id semantics \
                 are not reproduced, so this is not a parity claim",
                ModelType::ALL,
                ModelType::BERT_FAMILY_SIBLINGS
            ),
            Self::Unknown {
                value,
                source: ModelTypeSource::OverrideFlag,
            } => write!(
                f,
                "invalid --model-type value {value:?}; expected one of {:?}. A BERT-family \
                 sibling checkpoint (config.json model_type one of {:?}) is driven by passing \
                 --model-type bert, not by naming the sibling itself — its own declared name \
                 survives in the report's config_model_type field",
                ModelType::ALL,
                ModelType::BERT_FAMILY_SIBLINGS
            ),
            Self::Absent => write!(
                f,
                "config.json has no model_type field; pass --model-type explicitly, one of {:?}",
                ModelType::ALL
            ),
            Self::OverrideContradicts {
                file_type,
                override_type,
            } => write!(
                f,
                "config.json declares model_type {file_type:?}, one of this tier's own {:?}, \
                 but --model-type was passed as {override_type:?}; --model-type may only \
                 resolve a model_type this tier does not already recognize as one of its own \
                 three — an absent field, a BERT-family sibling such as {:?}, or another value \
                 entirely — it cannot relabel a checkpoint that already declares itself \
                 {file_type:?}. Pass a config.json-consistent value, or omit --model-type to \
                 detect {file_type:?} from the file",
                ModelType::ALL,
                ModelType::BERT_FAMILY_SIBLINGS
            ),
            Self::UnsupportedTensorPrefix { prefix } => write!(
                f,
                "checkpoint tensor keys are prefixed {prefix:?}, which the Bert builder's own \
                 two-layout probe does not know how to load — it loads only a \"bert.\"-wrapped \
                 BertForX checkpoint or an unprefixed root BertModel checkpoint. A BERT-family \
                 sibling saved through its own save_pretrained (e.g. RobertaForX) emits keys \
                 under its own class name, a third layout this tier cannot drive through \
                 --model-type bert; widening jammi-encoders' bert.rs prefix probe to accept \
                 sibling prefixes is a known follow-up, not something this tier can work around."
            ),
        }
    }
}

impl std::error::Error for ModelTypeError {}

/// [`resolve_target_modules`] found no universal default for `model_type`
/// and no explicit `--target-modules` was supplied. Typed rather than a
/// `format!().into()` string — its siblings in this module
/// ([`NoTargetModulesGivenError`], [`VocabTooSmallError`],
/// [`ModelTypeError`]) are already typed, so a caller that wants this
/// refusal's semantic surface (as opposed to just displaying it) can match
/// a variant instead of substring-matching a message that could drift.
#[derive(Debug)]
pub struct TargetModulesRequiredError {
    pub model_type: ModelType,
}

impl std::fmt::Display for TargetModulesRequiredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--target-modules is required for model_type {}: it has no universal default LoRA \
             target set; choose from {:?}",
            self.model_type,
            self.model_type.lora_target_vocabulary()
        )
    }
}

impl std::error::Error for TargetModulesRequiredError {}

/// Resolve the LoRA target-module list for `model_type`, applying the
/// tier's default policy: ModernBERT keeps its historical `Wqkv,Wo,Wi`
/// default when the caller passes none; BERT and DistilBERT have no
/// universal default LoRA target set (their linears are named differently,
/// and a borrowed ModernBERT default would silently match nothing on
/// either), so an explicit `--target-modules` is required for them and its
/// absence is a typed error naming the model's own selector vocabulary
/// rather than a refusal that only shows up later as "matched nothing".
fn resolve_target_modules(
    model_type: ModelType,
    explicit: Option<&[String]>,
) -> Result<Vec<String>, TargetModulesRequiredError> {
    if let Some(modules) = explicit {
        return Ok(modules.to_vec());
    }
    match model_type {
        ModelType::ModernBert => Ok(vec!["Wqkv".to_string(), "Wo".to_string(), "Wi".to_string()]),
        other => Err(TargetModulesRequiredError { model_type: other }),
    }
}

/// The `--target-modules` flag was passed but names no selector at all —
/// every comma-separated token was empty (`""`, `","`, `" , "`, ...).
/// Typed so this is caught at flag-parse time, before a model even loads:
/// an OMITTED flag defers to [`resolve_target_modules`]'s per-model default
/// policy (a legitimate, different outcome), but a flag that IS present and
/// resolves to zero selectors would otherwise sail through as `Some(vec![])`
/// — `resolve_target_modules` treats any `Some` as "the caller's explicit
/// choice" and returns it verbatim, so an empty explicit list would only
/// surface later as `run`'s "no trainable LoRA tensors — target_modules
/// matched nothing", after the checkpoint and its weights were already
/// loaded. That is the wrong failure to see for a typo in a comma list.
#[derive(Debug)]
pub struct NoTargetModulesGivenError;

impl std::fmt::Display for NoTargetModulesGivenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--target-modules was passed but names no selector (every comma-separated token \
             was empty); omit the flag entirely to use the model's default policy, or pass at \
             least one selector"
        )
    }
}

impl std::error::Error for NoTargetModulesGivenError {}

/// Parse the CLI's comma-separated `--target-modules` value into a selector
/// list. Call only when the flag was actually passed (an omitted flag stays
/// `None` at the call site and is never routed through here) — `raw` being
/// present-but-empty or all-commas is [`NoTargetModulesGivenError`], not an
/// empty `Vec`, so the refusal fires before the checkpoint loads rather than
/// after (see [`NoTargetModulesGivenError`]'s doc).
pub fn parse_target_modules_flag(raw: &str) -> Result<Vec<String>, NoTargetModulesGivenError> {
    let modules: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if modules.is_empty() {
        return Err(NoTargetModulesGivenError);
    }
    Ok(modules)
}

/// Parameters the tier drives its step off of.
#[derive(Debug, Clone)]
pub struct FinetuneStepParams {
    /// Directory holding `config.json` + `model.safetensors`.
    pub model_dir: std::path::PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub steps: usize,
    pub warmup: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f32,
    /// Comma-split LoRA target selectors, or `None` to apply the tier's
    /// per-model default policy (see [`resolve_target_modules`]): ModernBERT
    /// defaults to `Wqkv,Wo,Wi`; BERT and DistilBERT have no universal
    /// default and this tier refuses rather than silently matching nothing.
    pub target_modules: Option<Vec<String>>,
    /// Explicit override for the model family — required for a checkpoint
    /// whose `config.json` has no `model_type` field, and also the way to
    /// drive a BERT-family sibling (`roberta`/`camembert`/`xlm-roberta`)
    /// this tier does not autodetect (its own declared name then survives
    /// in the report's `config_model_type`). `None` detects the family from
    /// the file (see [`ModelType::detect`]).
    pub model_type_override: Option<String>,
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    /// CUDA ordinal, or `None` for CPU.
    pub cuda_device: Option<usize>,
    pub seed: u64,
    /// Encode anchor/positive/negative in ONE forward (what the trainer does)
    /// rather than three. Kept switchable because the difference between the two
    /// is the single largest term in this step on a dispatch-bound device, so
    /// the tier has to be able to measure it as a within-run A/B on one box
    /// rather than across binaries.
    pub batched_forward: bool,
}

/// A `vocab_size` too small to draw a non-pad synthetic id from. Typed so
/// the failure is a refusal, not the `% (vocab - 1)` divide-by-zero panic a
/// `vocab_size` of 0 or 1 would otherwise produce inside [`synthetic_ids`].
#[derive(Debug)]
pub struct VocabTooSmallError {
    pub vocab_size: usize,
}

impl std::fmt::Display for VocabTooSmallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "config.json vocab_size {} is too small for a synthetic non-pad id: need at least \
             2 (id 0 is reserved as pad, so a vocab of 1 has no other id to draw)",
            self.vocab_size
        )
    }
}

impl std::error::Error for VocabTooSmallError {}

/// Refuse BEFORE constructing the `Bert` encoder when `weights`' tensor
/// keys carry a prefix the `Bert` builder's own two-layout probe
/// (`jammi_encoders::bert.rs:508`, `"bert."`-wrapped or unprefixed root)
/// does not know. A BERT-family sibling checkpoint saved through its own
/// save_pretrained (`RobertaForX`, etc.) emits keys under ITS OWN class
/// name (`roberta.embeddings...`) — a THIRD layout, neither arm that probe
/// covers — so without this check `--model-type bert` on such a checkpoint
/// would reach `BertEmbeddings::load` and fail with a generic, untyped
/// "cannot find tensor" that names neither the mismatch nor the fix. This
/// tier is not `jammi-encoders` and cannot widen that probe itself (see
/// this module's top-level doc); it can only refuse typed, before the
/// misleading error, naming the prefix it actually found.
///
/// `None` (no tensor ending in the embeddings-weight suffix at all) is
/// deliberately let through unblocked: that is not the sibling-prefix case
/// this check exists for, and the encoder's own build will surface
/// whatever is actually wrong with a checkpoint that lacks the tensor
/// entirely.
fn probe_bert_tensor_prefix(weights: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    const EMBEDDING_SUFFIX: &str = "embeddings.word_embeddings.weight";
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(weights) }?;
    let found_prefix = mapped
        .tensors()
        .into_iter()
        .find_map(|(name, _)| name.strip_suffix(EMBEDDING_SUFFIX).map(str::to_string));
    match found_prefix.as_deref() {
        None | Some("") | Some("bert.") => Ok(()),
        Some(other) => Err(ModelTypeError::UnsupportedTensorPrefix {
            prefix: other.to_string(),
        }
        .into()),
    }
}

/// Deterministic synthetic token ids, uniform over `[1, vocab)` so no id is the
/// pad id. An LCG rather than a dependency, and identical across runs so two
/// measurements differ only in the code under test.
fn synthetic_ids(
    batch: usize,
    seq: usize,
    vocab: usize,
    seed: u64,
    device: &Device,
) -> Result<Tensor, VocabTooSmallError> {
    if vocab < 2 {
        return Err(VocabTooSmallError { vocab_size: vocab });
    }
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let ids: Vec<u32> = (0..batch * seq)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + ((s >> 33) as usize % (vocab - 1)) as u32
        })
        .collect();
    Ok(Tensor::from_vec(ids, (batch, seq), device).expect("synthetic ids"))
}

/// Run the tier and return its report block.
pub fn run(params: &FinetuneStepParams) -> Result<FinetuneStepTier, Box<dyn std::error::Error>> {
    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
    };
    let device_label = match params.cuda_device {
        Some(o) => format!("cuda:{o}"),
        None => "cpu".to_string(),
    };

    let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_raw)?;
    let detected = ModelType::detect(&config_json, params.model_type_override.as_deref())?;
    let model_type = detected.model_type;
    let config_model_type = detected.config_model_type;
    let model_type_override = detected.model_type_override;
    let target_modules = resolve_target_modules(model_type, params.target_modules.as_deref())?;
    let weights = params.model_dir.join("model.safetensors");
    if model_type == ModelType::Bert {
        probe_bert_tensor_prefix(&weights)?;
    }

    let varmap = VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &target_modules,
        layers_to_transform: &None,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        use_rslora: false,
        lora_dropout: (params.lora_dropout > 0.0).then_some(params.lora_dropout),
        rank_pattern: &empty_ranks,
        init_mode: jammi_lora::LoraInitMode::ZerosB,
        seed: params.seed,
    };
    let backbone_dtype = jammi_encoders::compute_precision_to_dtype(params.backbone_dtype);

    // Pooling is Mean for all three, mirroring the trainer's own dispatch
    // (`build_encoder_adapters` never calls `.pooling(..)`, so it takes each
    // builder's default — which is Mean; set explicitly here so this stays
    // true even if that default ever changes).
    let (mut encoder, vocab_size): (jammi_encoders::AnyEncoder, usize) = match model_type {
        ModelType::ModernBert => {
            let config: jammi_encoders::ModernBertConfig = serde_json::from_value(config_json)?;
            let vocab_size = config.vocab_size;
            let encoder = jammi_encoders::ModernBert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(backbone_dtype)
                .lora(lora)
                .build(&[weights.as_path()], &config, &device, &varmap)?;
            (jammi_encoders::AnyEncoder::ModernBert(encoder), vocab_size)
        }
        ModelType::Bert => {
            let config: jammi_encoders::BertConfig = serde_json::from_value(config_json)?;
            let vocab_size = config.vocab_size;
            let encoder = jammi_encoders::Bert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(backbone_dtype)
                .lora(lora)
                .build(&[weights.as_path()], &config, &device, &varmap)?;
            (jammi_encoders::AnyEncoder::Bert(encoder), vocab_size)
        }
        ModelType::DistilBert => {
            let config: jammi_encoders::DistilBertConfig = serde_json::from_value(config_json)?;
            let vocab_size = config.vocab_size;
            let encoder = jammi_encoders::DistilBert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(backbone_dtype)
                .lora(lora)
                .build(&[weights.as_path()], &config, &device, &varmap)?;
            (jammi_encoders::AnyEncoder::DistilBert(encoder), vocab_size)
        }
    };
    encoder.set_training(true);

    let trainable = varmap.all_vars();
    if trainable.is_empty() {
        return Err("no trainable LoRA tensors — target_modules matched nothing".into());
    }
    let mut opt = candle_nn::AdamW::new(
        trainable.clone(),
        candle_nn::ParamsAdamW {
            lr: 2e-4,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    let mask = Tensor::ones((params.batch, params.seq), DType::U32, &device)?;
    let blocks: Vec<Tensor> = (0..3)
        .map(|i| {
            synthetic_ids(
                params.batch,
                params.seq,
                vocab_size,
                params.seed + i,
                &device,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let vram_baseline = device_memory_used_bytes().unwrap_or(0);
    let sampler = VramSampler::start();

    // Positive-proof channel for the fused-vs-eager LayerNorm A/B: a
    // delta over the process-wide dispatch counters taken immediately
    // around the step loop, so this run's dispatch count is isolated
    // from anything an earlier tier in the same process invocation did.
    let ln_dispatch_before = jammi_encoders::ln_dispatch_snapshot();
    // Same mechanism, for the C3 fused RoPE kernel.
    let rope_dispatch_before = jammi_encoders::rope_dispatch_snapshot();
    // Same mechanism, for the C4 fused masked-softmax kernel.
    let softmax_dispatch_before = jammi_encoders::softmax_dispatch_snapshot();
    // Same mechanism, for the C5 fused GeGLU kernel.
    let geglu_dispatch_before = jammi_encoders::geglu_dispatch_snapshot();
    // Same mechanism, for the C6 fused LoRA-site epilogue.
    let lora_epilogue_dispatch_before = jammi_lora::lora_epilogue_dispatch_snapshot();

    let mut times = Vec::with_capacity(params.steps);
    for step in 0..(params.warmup + params.steps) {
        let t0 = Instant::now();
        let (a, p, n) = if params.batched_forward {
            // One forward over the concatenated groups, split after pooling —
            // the trainer's `encode_groups` shape.
            let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0)?;
            let joined_mask = Tensor::cat(&[&mask, &mask, &mask], 0)?;
            let all = encoder.forward(&joined, &joined_mask)?;
            let b = params.batch;
            (
                all.narrow(0, 0, b)?,
                all.narrow(0, b, b)?,
                all.narrow(0, 2 * b, b)?,
            )
        } else {
            (
                encoder.forward(&blocks[0], &mask)?,
                encoder.forward(&blocks[1], &mask)?,
                encoder.forward(&blocks[2], &mask)?,
            )
        };
        let loss = triplet_loss(&a, &p, &n, 0.3)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
        // Force completion before stopping the clock: candle's CUDA queue is
        // asynchronous, so without this the measured time is submission time,
        // not execution time — the classic way a GPU benchmark reports a number
        // far better than the work it did. Cast first: the loss carries the
        // backbone dtype, and reading a BF16 tensor as f32 is an error, not a
        // conversion.
        let _ = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        if step >= params.warmup {
            times.push(t0.elapsed().as_secs_f64());
        }
    }

    let ln_dispatch_after = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_after = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_after = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_after = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_after = jammi_lora::lora_epilogue_dispatch_snapshot();

    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    Ok(FinetuneStepTier {
        device: device_label,
        device_name: device_name(params.cuda_device),
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        model_type: model_type.as_str().to_string(),
        config_model_type,
        model_type_override,
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_dropout: params.lora_dropout as f64,
        target_modules,
        batched_forward: params.batched_forward,
        trainable_tensors: trainable.len(),
        steps_measured: times.len(),
        ln_fused_dispatches: ln_dispatch_after
            .fused
            .saturating_sub(ln_dispatch_before.fused),
        ln_eager_dispatches: ln_dispatch_after
            .eager
            .saturating_sub(ln_dispatch_before.eager),
        rope_fused_dispatches: rope_dispatch_after
            .fused
            .saturating_sub(rope_dispatch_before.fused),
        rope_eager_dispatches: rope_dispatch_after
            .eager
            .saturating_sub(rope_dispatch_before.eager),
        softmax_fused_dispatches: softmax_dispatch_after
            .fused
            .saturating_sub(softmax_dispatch_before.fused),
        softmax_eager_dispatches: softmax_dispatch_after
            .eager
            .saturating_sub(softmax_dispatch_before.eager),
        geglu_fused_dispatches: geglu_dispatch_after
            .fused
            .saturating_sub(geglu_dispatch_before.fused),
        geglu_eager_dispatches: geglu_dispatch_after
            .eager
            .saturating_sub(geglu_dispatch_before.eager),
        lora_epilogue_fused_dispatches: lora_epilogue_dispatch_after
            .fused
            .saturating_sub(lora_epilogue_dispatch_before.fused),
        lora_epilogue_eager_dispatches: lora_epilogue_dispatch_after
            .eager
            .saturating_sub(lora_epilogue_dispatch_before.eager),
        s_per_step_p50: Measurement::measured(p50, "s"),
        s_per_step_mean: Measurement::measured(mean, "s"),
        steps_per_s: Measurement::measured(1.0 / p50, "steps/s"),
        triplets_per_s: Measurement::measured(params.batch as f64 / p50, "triplets/s"),
        peak_rss_bytes: peak_rss_bytes(),
        peak_vram_bytes: match sampler {
            Some(s) => s.finish(vram_baseline),
            None => Measurement::not_yet_measured("bytes"),
        },
    })
}

/// The concrete device sub-class, so a recorded rate stays interpretable across
/// a heterogeneous rented fleet.
fn device_name(cuda_device: Option<usize>) -> String {
    match cuda_device {
        None => "cpu".to_string(),
        Some(_) => std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().lines().next().unwrap_or("unknown").to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    }
}

/// Peak resident set from `/proc/self/status` `VmHWM`. `None` off Linux, where
/// the field does not exist — recorded as absent rather than as a faked zero.
fn peak_rss_bytes() -> Measurement {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return Measurement::not_yet_measured("bytes");
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
            {
                return Measurement::measured(kb * 1024.0, "bytes");
            }
        }
    }
    Measurement::not_yet_measured("bytes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_bert_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_bert")
    }

    /// Write a tiny synthetic DistilBERT checkpoint (config.json +
    /// model.safetensors) to a fresh temp dir, mirroring
    /// `jammi-encoders/tests/it/distilbert.rs`'s `write_synthetic_weights` —
    /// no committed DistilBERT fixture exists in this workspace, and
    /// generating one in-test keeps the fixture generic/synthetic rather
    /// than shaped to any consumer's data.
    ///
    /// `layers` is a caller-supplied parameter, not a hardcoded `1`: a
    /// single-layer checkpoint cannot distinguish a per-layer selector
    /// mismatch from a global one (index 0 is even under any odd/even
    /// mutant), so every call site below passes `3` — odd, even, and
    /// last-layer skips are all visible in the resulting
    /// `trainable_tensors` count.
    fn write_tiny_distilbert(device: &Device, layers: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let hidden = 16usize;
        let inter = 32usize;
        let vocab = 64usize;
        let max_pos = 32usize;

        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "distilbert",
                "dim": hidden,
                "n_layers": layers,
                "n_heads": 2,
                "hidden_dim": inter,
                "vocab_size": vocab,
                "max_position_embeddings": max_pos,
            })
            .to_string(),
        )
        .expect("write config.json");

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let randn =
            |shape: (usize, usize)| Tensor::randn(0f32, 0.02, shape, device).expect("randn 2-D");
        let randn_1d = |n: usize| Tensor::randn(0f32, 0.02, (n,), device).expect("randn 1-D");
        let ones_1d = |n: usize| Tensor::ones((n,), DType::F32, device).expect("ones 1-D");
        let zeros_1d = |n: usize| Tensor::zeros((n,), DType::F32, device).expect("zeros 1-D");

        tensors.insert(
            "distilbert.embeddings.word_embeddings.weight".into(),
            randn((vocab, hidden)),
        );
        tensors.insert(
            "distilbert.embeddings.position_embeddings.weight".into(),
            randn((max_pos, hidden)),
        );
        tensors.insert(
            "distilbert.embeddings.LayerNorm.weight".into(),
            ones_1d(hidden),
        );
        tensors.insert(
            "distilbert.embeddings.LayerNorm.bias".into(),
            zeros_1d(hidden),
        );

        for n in 0..layers {
            let prefix = format!("distilbert.transformer.layer.{n}");
            for lin in ["q_lin", "k_lin", "v_lin", "out_lin"] {
                tensors.insert(
                    format!("{prefix}.attention.{lin}.weight"),
                    randn((hidden, hidden)),
                );
                tensors.insert(format!("{prefix}.attention.{lin}.bias"), randn_1d(hidden));
            }
            tensors.insert(format!("{prefix}.sa_layer_norm.weight"), ones_1d(hidden));
            tensors.insert(format!("{prefix}.sa_layer_norm.bias"), zeros_1d(hidden));

            tensors.insert(format!("{prefix}.ffn.lin1.weight"), randn((inter, hidden)));
            tensors.insert(format!("{prefix}.ffn.lin1.bias"), randn_1d(inter));
            tensors.insert(format!("{prefix}.ffn.lin2.weight"), randn((hidden, inter)));
            tensors.insert(format!("{prefix}.ffn.lin2.bias"), randn_1d(hidden));

            tensors.insert(
                format!("{prefix}.output_layer_norm.weight"),
                ones_1d(hidden),
            );
            tensors.insert(format!("{prefix}.output_layer_norm.bias"), zeros_1d(hidden));
        }

        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))
            .expect("save safetensors");
        dir
    }

    /// Write a tiny synthetic BERT-shaped checkpoint (config.json +
    /// model.safetensors) whose tensor keys carry `tensor_prefix` and whose
    /// `config.json` declares `config_model_type` — the one generator every
    /// BERT-layout fixture below goes through, so the tensor shapes and
    /// per-layer wiring cannot drift between the layouts that share them.
    ///
    /// `tensor_prefix` distinguishes the two layouts the `Bert` builder's
    /// own probe knows (`bert.rs:508`): `""` is a raw, unprefixed
    /// `BertModel` checkpoint; `"bert."` is a `BertForX`-wrapped one. A
    /// THIRD value — any other class name followed by `.`, e.g. `"roberta."`
    /// — produces a checkpoint shaped exactly like what a BERT-family
    /// sibling's own save_pretrained emits, which the `Bert` builder's
    /// probe does NOT know (see [`ModelType::detect`]'s doc and
    /// `ModelTypeError::UnsupportedTensorPrefix`).
    ///
    /// `layers` is caller-supplied, not hardcoded: a single-layer checkpoint
    /// cannot distinguish a per-layer selector mismatch from a global one
    /// (index 0 is even under any odd/even mutant), so every call site below
    /// passes `3` — odd, even, and last-layer skips are all visible in the
    /// resulting `trainable_tensors` count.
    fn write_tiny_bert_checkpoint(
        device: &Device,
        layers: usize,
        tensor_prefix: &str,
        config_model_type: &str,
    ) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let hidden = 16usize;
        let inter = 32usize;
        let vocab = 64usize;
        let max_pos = 32usize;
        let heads = 2usize;
        let type_vocab_size = 2usize;

        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": config_model_type,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "max_position_embeddings": max_pos,
                "type_vocab_size": type_vocab_size,
            })
            .to_string(),
        )
        .expect("write config.json");

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let randn =
            |shape: (usize, usize)| Tensor::randn(0f32, 0.02, shape, device).expect("randn 2-D");
        let randn_1d = |n: usize| Tensor::randn(0f32, 0.02, (n,), device).expect("randn 1-D");
        let ones_1d = |n: usize| Tensor::ones((n,), DType::F32, device).expect("ones 1-D");
        let zeros_1d = |n: usize| Tensor::zeros((n,), DType::F32, device).expect("zeros 1-D");

        tensors.insert(
            format!("{tensor_prefix}embeddings.word_embeddings.weight"),
            randn((vocab, hidden)),
        );
        tensors.insert(
            format!("{tensor_prefix}embeddings.position_embeddings.weight"),
            randn((max_pos, hidden)),
        );
        tensors.insert(
            format!("{tensor_prefix}embeddings.token_type_embeddings.weight"),
            randn((type_vocab_size, hidden)),
        );
        tensors.insert(
            format!("{tensor_prefix}embeddings.LayerNorm.weight"),
            ones_1d(hidden),
        );
        tensors.insert(
            format!("{tensor_prefix}embeddings.LayerNorm.bias"),
            zeros_1d(hidden),
        );

        for n in 0..layers {
            let prefix = format!("{tensor_prefix}encoder.layer.{n}");
            for lin in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
            ] {
                tensors.insert(format!("{prefix}.{lin}.weight"), randn((hidden, hidden)));
                tensors.insert(format!("{prefix}.{lin}.bias"), randn_1d(hidden));
            }
            tensors.insert(
                format!("{prefix}.attention.output.dense.weight"),
                randn((hidden, hidden)),
            );
            tensors.insert(
                format!("{prefix}.attention.output.dense.bias"),
                randn_1d(hidden),
            );
            tensors.insert(
                format!("{prefix}.attention.output.LayerNorm.weight"),
                ones_1d(hidden),
            );
            tensors.insert(
                format!("{prefix}.attention.output.LayerNorm.bias"),
                zeros_1d(hidden),
            );

            tensors.insert(
                format!("{prefix}.intermediate.dense.weight"),
                randn((inter, hidden)),
            );
            tensors.insert(format!("{prefix}.intermediate.dense.bias"), randn_1d(inter));

            tensors.insert(
                format!("{prefix}.output.dense.weight"),
                randn((hidden, inter)),
            );
            tensors.insert(format!("{prefix}.output.dense.bias"), randn_1d(hidden));
            tensors.insert(format!("{prefix}.output.LayerNorm.weight"), ones_1d(hidden));
            tensors.insert(format!("{prefix}.output.LayerNorm.bias"), zeros_1d(hidden));
        }

        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))
            .expect("save safetensors");
        dir
    }

    /// A `"bert."`-wrapped `BertForX`-layout checkpoint. The committed
    /// `tiny_bert` fixture (`cookbook/fixtures/tiny_bert`) is the OTHER
    /// layout — a raw `BertModel` checkpoint with unprefixed keys — so this
    /// generator is what drives the `"bert."`-prefixed arm
    /// (`bert.rs:508`'s `contains_tensor` probe) end-to-end; that arm failed
    /// live before this fixture existed.
    fn write_tiny_prefixed_bert(device: &Device, layers: usize) -> tempfile::TempDir {
        write_tiny_bert_checkpoint(device, layers, "bert.", "bert")
    }

    /// An unprefixed root `BertModel`-layout checkpoint, generated
    /// synthetically (rather than reused from the committed `tiny_bert`
    /// fixture) so it can carry `layers >= 2` without touching a fixture
    /// shared by the rest of the workspace.
    fn write_tiny_bert_unprefixed(device: &Device, layers: usize) -> tempfile::TempDir {
        write_tiny_bert_checkpoint(device, layers, "", "bert")
    }

    /// A BERT-family-sibling checkpoint laid out exactly the way that
    /// sibling's OWN save_pretrained would emit it — tensor keys prefixed
    /// with the sibling's own class name (`"roberta."`), config.json
    /// declaring `model_type: "roberta"`. This is the THIRD layout
    /// `ModelType::BERT_FAMILY_SIBLINGS`' doc describes and the `Bert`
    /// builder's two-layout probe does not know — see
    /// `finetune_step_end_to_end_cpu_smoke_bert_sibling_prefix_is_a_typed_refusal`.
    fn write_tiny_roberta_prefixed(device: &Device, layers: usize) -> tempfile::TempDir {
        write_tiny_bert_checkpoint(device, layers, "roberta.", "roberta")
    }

    /// Write a tiny synthetic ModernBERT checkpoint (config.json +
    /// model.safetensors) under the HuggingFace `"model."`-prefixed
    /// tensor-key convention `jammi_encoders::ModernBert::builder().build`
    /// reads (`modernbert.rs`: `model.embeddings.tok_embeddings`,
    /// `model.layers.{n}.attn.{Wqkv,Wo}`, `model.layers.{n}.mlp.{Wi,Wo}`,
    /// `model.final_norm`, plus a per-layer `attn_norm`/`mlp_norm` and a
    /// layer-0-only-absent `attn_norm` — see `ModernBertBuilder::build`).
    /// No committed ModernBERT fixture in this workspace carries more than
    /// one layer, so this generator (not the committed `tiny_modernbert`)
    /// is what the exact-`trainable_tensors` smoke below drives.
    /// `global_attn_every_n_layers: 1` keeps every layer on the same RoPE
    /// table, since which RoPE table a layer uses does not bear on LoRA-site
    /// selection — the property this generator exists to pin.
    fn write_tiny_modernbert(device: &Device, layers: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let hidden = 16usize;
        let inter = 32usize;
        let vocab = 64usize;
        let max_pos = 32usize;
        let heads = 2usize;

        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "modernbert",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "max_position_embeddings": max_pos,
                "layer_norm_eps": 1e-5,
                "pad_token_id": 0,
                "global_attn_every_n_layers": 1,
                "global_rope_theta": 160000.0,
                "local_attention": max_pos,
                "local_rope_theta": 10000.0,
            })
            .to_string(),
        )
        .expect("write config.json");

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let randn =
            |shape: (usize, usize)| Tensor::randn(0f32, 0.02, shape, device).expect("randn 2-D");
        let ones_1d = |n: usize| Tensor::ones((n,), DType::F32, device).expect("ones 1-D");

        tensors.insert(
            "model.embeddings.tok_embeddings.weight".into(),
            randn((vocab, hidden)),
        );
        tensors.insert("model.embeddings.norm.weight".into(), ones_1d(hidden));

        for n in 0..layers {
            let prefix = format!("model.layers.{n}");
            tensors.insert(
                format!("{prefix}.attn.Wqkv.weight"),
                randn((hidden * 3, hidden)),
            );
            tensors.insert(format!("{prefix}.attn.Wo.weight"), randn((hidden, hidden)));
            if n != 0 {
                tensors.insert(format!("{prefix}.attn_norm.weight"), ones_1d(hidden));
            }
            tensors.insert(
                format!("{prefix}.mlp.Wi.weight"),
                randn((inter * 2, hidden)),
            );
            tensors.insert(format!("{prefix}.mlp.Wo.weight"), randn((hidden, inter)));
            tensors.insert(format!("{prefix}.mlp_norm.weight"), ones_1d(hidden));
        }
        tensors.insert("model.final_norm.weight".into(), ones_1d(hidden));

        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))
            .expect("save safetensors");
        dir
    }

    fn base_params(model_dir: std::path::PathBuf) -> FinetuneStepParams {
        FinetuneStepParams {
            model_dir,
            batch: 2,
            seq: 6,
            steps: 1,
            warmup: 0,
            lora_rank: 2,
            lora_alpha: 4.0,
            lora_dropout: 0.0,
            target_modules: None,
            model_type_override: None,
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            cuda_device: None,
            seed: 1,
            batched_forward: true,
        }
    }

    #[test]
    fn detects_the_three_positive_model_types() {
        for (raw, expected) in [
            ("modernbert", ModelType::ModernBert),
            ("bert", ModelType::Bert),
            ("distilbert", ModelType::DistilBert),
        ] {
            let json = serde_json::json!({ "model_type": raw });
            let detected = ModelType::detect(&json, None).unwrap();
            assert_eq!(detected.model_type, expected);
            // B2: `config_model_type` carries the file's own declaration
            // EXACTLY as read, even though it already agrees with the
            // resolved model — "declared and agreed" is a fact the record
            // preserves, not a state collapsed to `None`. No `--model-type`
            // was passed, so `model_type_override` is `None` — the OPPOSITE
            // of `absent_model_type_requires_explicit_override`'s override
            // branch below, where `config_model_type` is `None` (nothing
            // was declared) and `model_type_override` is `Some(_)`
            // (everything came from the flag). A single collapsed
            // `Option<String>` could not tell these two apart.
            assert_eq!(detected.config_model_type.as_deref(), Some(raw));
            assert_eq!(detected.model_type_override, None);
        }
    }

    #[test]
    fn unknown_model_type_is_a_typed_error_naming_all_three_and_the_accepted_siblings() {
        let json = serde_json::json!({ "model_type": "gpt2" });
        let err = ModelType::detect(&json, None).unwrap_err();
        assert!(matches!(
            &err,
            ModelTypeError::Unknown { value, source: ModelTypeSource::ConfigJson } if value == "gpt2"
        ));
        let msg = err.to_string();
        for name in ModelType::ALL {
            assert!(msg.contains(name), "error {msg:?} must name {name}");
        }
        for sibling in ModelType::BERT_FAMILY_SIBLINGS {
            assert!(
                msg.contains(sibling),
                "error {msg:?} must name accepted sibling {sibling}"
            );
        }
    }

    /// B3: an INVALID `--model-type` value is refused as an invalid FLAG —
    /// typed `Unknown { source: OverrideFlag, .. }`, naming `--model-type`
    /// — even against a `config.json` that itself declares a perfectly
    /// valid `model_type` (`"bert"`). Before B3 this reached
    /// `OverrideContradicts` instead (comparing the unvalidated override
    /// string against the file's declaration first), producing a message
    /// that told the operator "config.json declares bert but --model-type
    /// was berrt" — true as prose, but the wrong surface: `berrt` is not a
    /// model type at all, on any surface, and the fix is to the FLAG, not
    /// to `config.json`.
    #[test]
    fn invalid_override_on_a_valid_bert_config_is_a_typed_error_naming_the_flag() {
        let json = serde_json::json!({ "model_type": "bert" });
        let err = ModelType::detect(&json, Some("berrt")).unwrap_err();
        assert!(matches!(
            &err,
            ModelTypeError::Unknown { value, source: ModelTypeSource::OverrideFlag } if value == "berrt"
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("--model-type"),
            "refusal {msg:?} must name the flag, not config.json"
        );
        assert!(
            msg.contains("berrt"),
            "refusal {msg:?} must name the bad value"
        );
    }

    #[test]
    fn absent_model_type_requires_explicit_override() {
        let json = serde_json::json!({});
        assert!(matches!(
            ModelType::detect(&json, None).unwrap_err(),
            ModelTypeError::Absent
        ));
        let detected = ModelType::detect(&json, Some("bert")).unwrap();
        assert_eq!(detected.model_type, ModelType::Bert);
        // B2: model_type absent + `--model-type bert` → `config_model_type`
        // is `None` (there is no file declaration to diverge from, or to
        // agree with), but `model_type_override` is `Some("bert")` — the
        // resolved family came ENTIRELY from the flag. This is the value
        // `detects_the_three_positive_model_types` above must now differ
        // from: there, the file declared something and no override was
        // given; here, the file declared nothing and the override did all
        // the work. Two distinguishable states, not one collapsed `None`.
        assert_eq!(detected.config_model_type, None);
        assert_eq!(detected.model_type_override.as_deref(), Some("bert"));
    }

    /// A real RoBERTa `config.json` carries `model_type = "roberta"`. This
    /// tier does not autodetect it (it is not one of [`ModelType::ALL`]),
    /// but `--model-type bert` is ACCEPTED — `"roberta"` is not one of this
    /// tier's own three types, so this is not the genuine-mislabel case
    /// [`ModelTypeError::OverrideContradicts`] exists for — and the
    /// resolved row is `Bert` with `config_model_type` carrying the file's
    /// own `"roberta"` string and `model_type_override` carrying the
    /// flag's own `"bert"`, so the row stays attributable to both what the
    /// checkpoint declares itself to be AND what resolved it.
    #[test]
    fn bert_family_sibling_config_accepts_bert_override_and_records_its_own_declaration() {
        let json = serde_json::json!({ "model_type": "roberta" });
        let detected = ModelType::detect(&json, Some("bert")).unwrap();
        assert_eq!(detected.model_type, ModelType::Bert);
        assert_eq!(detected.config_model_type.as_deref(), Some("roberta"));
        assert_eq!(detected.model_type_override.as_deref(), Some("bert"));
    }

    #[test]
    fn target_modules_refuses_silently_matching_nothing_for_bert_and_distilbert() {
        for mt in [ModelType::Bert, ModelType::DistilBert] {
            let err = resolve_target_modules(mt, None).unwrap_err();
            // A1: match on the typed variant, not just its rendered message.
            assert_eq!(err.model_type, mt);
            let msg = err.to_string();
            for name in mt.lora_target_vocabulary() {
                assert!(msg.contains(name), "refusal {msg:?} must name {name}");
            }
        }
        // ModernBERT keeps its historical default.
        assert_eq!(
            resolve_target_modules(ModelType::ModernBert, None).unwrap(),
            vec!["Wqkv".to_string(), "Wo".to_string(), "Wi".to_string()]
        );
        // An explicit list always wins, for every model.
        let explicit = vec!["query".to_string()];
        assert_eq!(
            resolve_target_modules(ModelType::Bert, Some(&explicit)).unwrap(),
            explicit
        );
    }

    /// The layer count every synthetic multi-layer fixture below is built
    /// at (see [`write_tiny_bert_checkpoint`]'s doc for why `1` cannot
    /// catch a per-layer selector mutation).
    const SMOKE_LAYERS: usize = 3;

    /// B1: the exact trainable-tensor count `target_modules` must produce
    /// on a `layers`-layer encoder, derived from the SAME
    /// `jammi_lora::should_apply_lora` vocabulary table the
    /// `*_selector_vocabulary_matches_real_module_names` tests below use —
    /// so this multiplication is pinned against the production matcher,
    /// not a hand-derived constant that could silently drift from it. Each
    /// matched site carries exactly 2 trainable tensors (LoRA A and B).
    fn expected_trainable_tensors(
        target_modules: &[&str],
        module_names: &[&str],
        layers: usize,
    ) -> usize {
        let matched_per_layer: usize = target_modules
            .iter()
            .map(|selector| sites_matched(selector, module_names))
            .sum();
        matched_per_layer * layers * 2
    }

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_bert() {
        let device = Device::Cpu;
        let dir = write_tiny_bert_unprefixed(&device, SMOKE_LAYERS);
        let mut params = base_params(dir.path().to_path_buf());
        let targets = ["query", "value"];
        params.target_modules = Some(targets.iter().map(|s| s.to_string()).collect());
        let tier = run(&params).expect("bert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "bert");
        assert_eq!(tier.steps_measured, 1);
        // B1: the mutation `should_apply_lora`'s odd-layer-skip would halve
        // this on a 1-layer fixture's layer 0 (even) not at all — only a
        // `layers >= 2` fixture with an EXACT (not `> 0`) assertion catches
        // it. See this module's mutation-verification note above `run`.
        assert_eq!(
            tier.trainable_tensors,
            expected_trainable_tensors(&targets, &bert_module_names(), SMOKE_LAYERS)
        );
    }

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_modernbert() {
        let device = Device::Cpu;
        let dir = write_tiny_modernbert(&device, SMOKE_LAYERS);
        let params = base_params(dir.path().to_path_buf()); // default target_modules
        let tier = run(&params).expect("modernbert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "modernbert");
        let default_targets = ["Wqkv", "Wo", "Wi"];
        assert_eq!(
            tier.trainable_tensors,
            expected_trainable_tensors(&default_targets, &modernbert_module_names(), SMOKE_LAYERS)
        );
    }

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_distilbert() {
        let device = Device::Cpu;
        let dir = write_tiny_distilbert(&device, SMOKE_LAYERS);
        let mut params = base_params(dir.path().to_path_buf());
        let targets = ["q_lin", "v_lin"];
        params.target_modules = Some(targets.iter().map(|s| s.to_string()).collect());
        let tier = run(&params).expect("distilbert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "distilbert");
        assert_eq!(
            tier.trainable_tensors,
            expected_trainable_tensors(&targets, &distilbert_module_names(), SMOKE_LAYERS)
        );
    }

    #[test]
    fn target_modules_refusal_fires_through_run_for_bert() {
        let dir = tiny_bert_dir();
        assert!(
            dir.join("config.json").exists(),
            "tiny_bert fixture must exist at {dir:?} — a missing fixture is a test failure, \
             not a silent skip"
        );
        let params = base_params(dir); // target_modules: None
        let err = run(&params).unwrap_err();
        // A1: match on the typed variant, downcast through `run`'s boxed
        // error — not a substring match on its rendered message.
        let typed = err
            .downcast_ref::<TargetModulesRequiredError>()
            .expect("refusal must be TargetModulesRequiredError, not an untyped string");
        assert_eq!(typed.model_type, ModelType::Bert);
    }

    /// The arm that failed live: a `"bert."`-prefixed `BertForX` checkpoint
    /// (`bert.rs:508`'s wrapped-layout probe), never previously exercised by
    /// any test in this workspace — `finetune_step_end_to_end_cpu_smoke_bert`
    /// only covers the unprefixed raw-`BertModel` layout.
    #[test]
    fn finetune_step_end_to_end_cpu_smoke_bert_prefixed_layout() {
        let device = Device::Cpu;
        let dir = write_tiny_prefixed_bert(&device, SMOKE_LAYERS);
        let mut params = base_params(dir.path().to_path_buf());
        let targets = ["query", "value"];
        params.target_modules = Some(targets.iter().map(|s| s.to_string()).collect());
        let tier = run(&params)
            .expect("bert (BertForX-prefixed layout) finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "bert");
        assert_eq!(tier.steps_measured, 1);
        assert_eq!(
            tier.trainable_tensors,
            expected_trainable_tensors(&targets, &bert_module_names(), SMOKE_LAYERS)
        );
        // Acceptance evidence for the prefixed-layout arm — printed (run
        // with `--nocapture`) so the run that exercises `bert.rs:508`'s
        // `contains_tensor("bert.embeddings...")` probe leaves a record of
        // what it actually built, not just a pass/fail bit.
        println!(
            "prefixed-BERT smoke: model_type={} trainable_tensors={} \
             lora_epilogue_fused_dispatches={} lora_epilogue_eager_dispatches={}",
            tier.model_type,
            tier.trainable_tensors,
            tier.lora_epilogue_fused_dispatches,
            tier.lora_epilogue_eager_dispatches,
        );
    }

    /// B4(a): a BERT-family sibling checkpoint saved through its OWN
    /// save_pretrained layout (`"roberta."`-prefixed keys, matching what
    /// `RobertaForX` actually emits) refuses TYPED, before the encoder is
    /// even constructed, rather than failing deep inside `BertEmbeddings::load`
    /// with a generic "cannot find tensor" that names neither the mismatch
    /// nor the fix. This is a bench-side refusal, not a numerics fix:
    /// widening `jammi-encoders`' `bert.rs` prefix probe to accept sibling
    /// prefixes is the real generalisation, and is out of this crate's
    /// scope (see `ModelTypeError::UnsupportedTensorPrefix`'s doc).
    #[test]
    fn finetune_step_end_to_end_cpu_smoke_bert_sibling_prefix_is_a_typed_refusal() {
        let device = Device::Cpu;
        let dir = write_tiny_roberta_prefixed(&device, SMOKE_LAYERS);
        let mut params = base_params(dir.path().to_path_buf());
        params.model_type_override = Some("bert".to_string());
        params.target_modules = Some(vec!["query".to_string(), "value".to_string()]);
        let err = run(&params).unwrap_err();
        let typed = err
            .downcast_ref::<ModelTypeError>()
            .expect("refusal must be ModelTypeError, not an untyped string");
        assert!(
            matches!(
                typed,
                ModelTypeError::UnsupportedTensorPrefix { prefix } if prefix == "roberta."
            ),
            "expected UnsupportedTensorPrefix{{prefix: \"roberta.\"}}, got {typed:?}"
        );
    }

    /// B4(b): the complement of the refusal above — a BERT-family sibling
    /// checkpoint declaring `model_type: "roberta"` but laid out in the
    /// UNPREFIXED root layout (the same layout `write_tiny_bert_unprefixed`
    /// produces) drives all the way through `run`, proving the probe does
    /// not false-positive on the layout the `Bert` builder actually
    /// supports — only the genuinely unknown `"roberta."` prefix above is
    /// refused.
    #[test]
    fn finetune_step_end_to_end_cpu_smoke_roberta_declared_unprefixed_layout_loads() {
        let device = Device::Cpu;
        let dir = write_tiny_bert_checkpoint(&device, SMOKE_LAYERS, "", "roberta");
        let mut params = base_params(dir.path().to_path_buf());
        params.model_type_override = Some("bert".to_string());
        let targets = ["query", "value"];
        params.target_modules = Some(targets.iter().map(|s| s.to_string()).collect());
        let tier = run(&params).expect(
            "roberta-declared, unprefixed-layout checkpoint loads through --model-type bert",
        );
        assert_eq!(tier.model_type, "bert");
        assert_eq!(tier.config_model_type.as_deref(), Some("roberta"));
        assert_eq!(tier.model_type_override.as_deref(), Some("bert"));
        assert_eq!(
            tier.trainable_tensors,
            expected_trainable_tensors(&targets, &bert_module_names(), SMOKE_LAYERS)
        );
    }

    #[test]
    fn override_contradicting_config_model_type_is_a_typed_error() {
        // `"distilbert"` is one of this tier's own three (`ModelType::ALL`)
        // — a genuine mislabel against a different override, unlike the
        // BERT-family-sibling case above.
        let json = serde_json::json!({ "model_type": "distilbert" });
        let err = ModelType::detect(&json, Some("bert")).unwrap_err();
        assert!(matches!(
            &err,
            ModelTypeError::OverrideContradicts { file_type, override_type }
                if file_type == "distilbert" && override_type == "bert"
        ));
        let msg = err.to_string();
        assert!(
            msg.contains("distilbert"),
            "error {msg:?} must name the file's model_type"
        );
        assert!(msg.contains("bert"), "error {msg:?} must name the override");

        // A matching override is a no-op, not a contradiction — this is the
        // ABSENT-field case the override actually exists for, extended to
        // "present and identical" rather than "present and different". B2:
        // `config_model_type` still carries the file's own declaration
        // (`Some("distilbert")`, not collapsed to `None` by the agreement),
        // and `model_type_override` records that the flag was passed too.
        let detected = ModelType::detect(&json, Some("distilbert")).unwrap();
        assert_eq!(detected.model_type, ModelType::DistilBert);
        assert_eq!(detected.config_model_type.as_deref(), Some("distilbert"));
        assert_eq!(detected.model_type_override.as_deref(), Some("distilbert"));
    }

    #[test]
    fn target_modules_flag_empty_or_all_commas_is_a_typed_parse_time_error() {
        for raw in ["", ",", " , ", ",,,"] {
            let err = parse_target_modules_flag(raw).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("--target-modules"),
                "refusal {msg:?} must name the flag"
            );
        }
        assert_eq!(
            parse_target_modules_flag("query, value").unwrap(),
            vec!["query".to_string(), "value".to_string()]
        );
    }

    #[test]
    fn synthetic_ids_refuses_vocab_size_below_two() {
        let device = Device::Cpu;
        for vocab in [0usize, 1usize] {
            let err = synthetic_ids(2, 4, vocab, 1, &device).unwrap_err();
            assert_eq!(err.vocab_size, vocab);
        }
        assert!(synthetic_ids(2, 4, 2, 1, &device).is_ok());
    }

    /// BERT's real per-layer module names — the FIRST argument to
    /// `LoraSite::build` (`bert.rs:536-548`), the string
    /// `jammi_lora::should_apply_lora` actually matches a selector against.
    /// Embedded here rather than read from `bert.rs`'s own (private)
    /// `lora_sites` helper: that helper returns the SECOND argument (the
    /// LoRA adapter subpath, used only for dropout-position bookkeeping) —
    /// reusing it for module-name matching would silently reintroduce this
    /// block's bug.
    fn bert_module_names() -> [&'static str; 6] {
        [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
    }

    /// DistilBERT's real per-layer module names (`distilbert.rs:477-533`).
    fn distilbert_module_names() -> [&'static str; 6] {
        ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    }

    /// ModernBERT's real per-layer module names (`modernbert.rs:1178-1215`).
    fn modernbert_module_names() -> [&'static str; 4] {
        ["Wqkv", "Wo", "Wi", "mlp.Wo"]
    }

    /// How many of `module_names` one advertised `selector` matches, via
    /// the SAME function the encoders call at build time
    /// (`jammi_lora::should_apply_lora`) rather than a re-implementation of
    /// its suffix-match rule — so this table cannot drift from the real
    /// matching semantics.
    fn sites_matched(selector: &str, module_names: &[&str]) -> usize {
        let targets = [selector.to_string()];
        module_names
            .iter()
            .filter(|name| jammi_lora::should_apply_lora(name, &targets, 0, &None))
            .count()
    }

    #[test]
    fn bert_selector_vocabulary_matches_real_module_names() {
        let names = bert_module_names();
        // (selector, sites it matches per layer under `should_apply_lora`).
        // "output.dense" is 2, not 1: `should_apply_lora`'s own suffix-match
        // rule (`module_name.ends_with(t)`) means "attention.output.dense"
        // also matches, since it ends with "output.dense" too. There is no
        // suffix-only selector that reaches the FFN `output.dense` without
        // also reaching `attention.output.dense` — that is the trainer's own
        // matching semantics, not a bug this tier disambiguates.
        let expected: &[(&str, usize)] = &[
            ("query", 1),
            ("key", 1),
            ("value", 1),
            ("attention.output.dense", 1),
            ("intermediate.dense", 1),
            ("output.dense", 2),
        ];
        let advertised: Vec<&str> = expected.iter().map(|(s, _)| *s).collect();
        assert_eq!(
            ModelType::Bert.lora_target_vocabulary(),
            advertised.as_slice()
        );
        for (selector, want) in expected {
            assert_eq!(
                sites_matched(selector, &names),
                *want,
                "selector {selector:?} must match {want} site(s) of {names:?}"
            );
        }
    }

    #[test]
    fn distilbert_selector_vocabulary_matches_real_module_names_one_to_one() {
        let names = distilbert_module_names();
        for selector in ModelType::DistilBert.lora_target_vocabulary() {
            assert_eq!(
                sites_matched(selector, &names),
                1,
                "selector {selector:?} must match exactly one of {names:?}"
            );
        }
    }

    #[test]
    fn modernbert_selector_vocabulary_matches_real_module_names() {
        let names = modernbert_module_names();
        // "Wo" is 2, not 1: "mlp.Wo" ends with "Wo" too, so the tier's
        // historical ModernBERT default reaches both `Wo` sites per layer —
        // documented (not "fixed") the same way as BERT's `output.dense`
        // ambiguity above, since it is the same `should_apply_lora`
        // suffix-match rule at work, and this IS the trainer's semantics.
        let expected: &[(&str, usize)] = &[("Wqkv", 1), ("Wo", 2), ("Wi", 1)];
        let advertised: Vec<&str> = expected.iter().map(|(s, _)| *s).collect();
        assert_eq!(
            ModelType::ModernBert.lora_target_vocabulary(),
            advertised.as_slice()
        );
        for (selector, want) in expected {
            assert_eq!(
                sites_matched(selector, &names),
                *want,
                "selector {selector:?} must match {want} site(s) of {names:?}"
            );
        }
    }
}
