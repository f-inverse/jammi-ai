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

/// Which of the three trainer-supported text encoders
/// (`jammi_ai::fine_tune::worker::build_encoder_adapters`) a checkpoint
/// directory holds.
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

impl ModelType {
    pub const ALL: [&'static str; 3] = ["modernbert", "bert", "distilbert"];

    fn parse(raw: &str) -> Result<Self, ModelTypeError> {
        match raw {
            "modernbert" => Ok(Self::ModernBert),
            "bert" => Ok(Self::Bert),
            "distilbert" => Ok(Self::DistilBert),
            other => Err(ModelTypeError::Unknown(other.to_string())),
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
    /// override always wins over the file's own `model_type` field — the
    /// override exists precisely because `model_type` CAN be absent from a
    /// real checkpoint: `jammi_encoders::BertConfig`'s own `model_type` field
    /// is `#[serde(default)]` `Option<String>` (a raw `BertModel` checkpoint
    /// need not carry it), and the trainer's own dispatch treats an absent
    /// field as `"bert"` rather than refusing to build.
    pub fn detect(
        config_json: &serde_json::Value,
        override_type: Option<&str>,
    ) -> Result<Self, ModelTypeError> {
        if let Some(explicit) = override_type {
            return Self::parse(explicit);
        }
        match config_json.get("model_type").and_then(|v| v.as_str()) {
            Some(raw) => Self::parse(raw),
            None => Err(ModelTypeError::Absent),
        }
    }

    /// The LoRA target-module selector vocabulary this model's encoder
    /// actually wires up — the short names each encoder's `LoraSite::build`
    /// call passes as `target_name` (`bert.rs`'s six per-layer linears,
    /// `distilbert.rs`'s six, `modernbert.rs`'s `Wqkv`/`Wo`/`Wi`/`mlp.Wo`).
    /// `jammi_lora::should_apply_lora` matches a target string against the
    /// END of the module name, so `"Wo"` also matches ModernBERT's `mlp.Wo`.
    pub fn lora_target_vocabulary(self) -> &'static [&'static str] {
        match self {
            Self::ModernBert => &["Wqkv", "Wo", "Wi"],
            Self::Bert => &[
                "query",
                "key",
                "value",
                "dense",
                "intermediate_dense",
                "output_dense",
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

/// Why [`ModelType::detect`] could not resolve a model family. Typed (not a
/// bare string) so a caller matching on the failure never has to
/// string-match a message.
#[derive(Debug)]
pub enum ModelTypeError {
    /// `model_type` was present but not one of [`ModelType::ALL`].
    Unknown(String),
    /// `config.json` has no `model_type` field and no `--model-type`
    /// override was supplied.
    Absent,
}

impl std::fmt::Display for ModelTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(raw) => write!(
                f,
                "unknown model_type {raw:?} in config.json; expected one of {:?}, or pass \
                 --model-type explicitly",
                ModelType::ALL
            ),
            Self::Absent => write!(
                f,
                "config.json has no model_type field; pass --model-type explicitly, one of {:?}",
                ModelType::ALL
            ),
        }
    }
}

impl std::error::Error for ModelTypeError {}

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
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let Some(modules) = explicit {
        return Ok(modules.to_vec());
    }
    match model_type {
        ModelType::ModernBert => Ok(vec!["Wqkv".to_string(), "Wo".to_string(), "Wi".to_string()]),
        other => Err(format!(
            "--target-modules is required for model_type {other}: it has no universal default \
             LoRA target set; choose from {:?}",
            other.lora_target_vocabulary()
        )
        .into()),
    }
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
    /// Explicit override for the model family, for a checkpoint whose
    /// `config.json` has no `model_type` field. `None` detects it from the
    /// file (see [`ModelType::detect`]).
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

/// Deterministic synthetic token ids, uniform over `[1, vocab)` so no id is the
/// pad id. An LCG rather than a dependency, and identical across runs so two
/// measurements differ only in the code under test.
fn synthetic_ids(batch: usize, seq: usize, vocab: usize, seed: u64, device: &Device) -> Tensor {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let ids: Vec<u32> = (0..batch * seq)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + ((s >> 33) as usize % (vocab - 1)) as u32
        })
        .collect();
    Tensor::from_vec(ids, (batch, seq), device).expect("synthetic ids")
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
    let model_type = ModelType::detect(&config_json, params.model_type_override.as_deref())?;
    let target_modules = resolve_target_modules(model_type, params.target_modules.as_deref())?;
    let weights = params.model_dir.join("model.safetensors");

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
        .collect();

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

    fn tiny_modernbert_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny_modernbert")
    }

    /// Write a tiny synthetic DistilBERT checkpoint (config.json +
    /// model.safetensors) to a fresh temp dir, mirroring
    /// `jammi-encoders/tests/it/distilbert.rs`'s `write_synthetic_weights` —
    /// no committed DistilBERT fixture exists in this workspace, and
    /// generating one in-test keeps the fixture generic/synthetic rather
    /// than shaped to any consumer's data.
    fn write_tiny_distilbert(device: &Device) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let hidden = 16usize;
        let inter = 32usize;
        let vocab = 64usize;
        let max_pos = 32usize;
        let layers = 1usize;

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
            assert_eq!(ModelType::detect(&json, None).unwrap(), expected);
        }
    }

    #[test]
    fn unknown_model_type_is_a_typed_error_naming_all_three() {
        let json = serde_json::json!({ "model_type": "gpt2" });
        let err = ModelType::detect(&json, None).unwrap_err();
        assert!(matches!(&err, ModelTypeError::Unknown(s) if s == "gpt2"));
        let msg = err.to_string();
        for name in ModelType::ALL {
            assert!(msg.contains(name), "error {msg:?} must name {name}");
        }
    }

    #[test]
    fn absent_model_type_requires_explicit_override() {
        let json = serde_json::json!({});
        assert!(matches!(
            ModelType::detect(&json, None).unwrap_err(),
            ModelTypeError::Absent
        ));
        assert_eq!(
            ModelType::detect(&json, Some("bert")).unwrap(),
            ModelType::Bert
        );
    }

    #[test]
    fn target_modules_refuses_silently_matching_nothing_for_bert_and_distilbert() {
        for mt in [ModelType::Bert, ModelType::DistilBert] {
            let err = resolve_target_modules(mt, None).unwrap_err();
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

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_bert() {
        let dir = tiny_bert_dir();
        if !dir.join("config.json").exists() {
            eprintln!("skipping: tiny_bert fixture not present at {dir:?}");
            return;
        }
        let mut params = base_params(dir);
        params.target_modules = Some(vec!["query".to_string(), "value".to_string()]);
        let tier = run(&params).expect("bert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "bert");
        assert!(tier.trainable_tensors > 0);
        assert_eq!(tier.steps_measured, 1);
    }

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_modernbert() {
        let dir = tiny_modernbert_dir();
        if !dir.join("config.json").exists() {
            eprintln!("skipping: tiny_modernbert fixture not present at {dir:?}");
            return;
        }
        let params = base_params(dir);
        let tier = run(&params).expect("modernbert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "modernbert");
        assert!(tier.trainable_tensors > 0);
    }

    #[test]
    fn finetune_step_end_to_end_cpu_smoke_distilbert() {
        let device = Device::Cpu;
        let dir = write_tiny_distilbert(&device);
        let mut params = base_params(dir.path().to_path_buf());
        params.target_modules = Some(vec!["q_lin".to_string(), "v_lin".to_string()]);
        let tier = run(&params).expect("distilbert finetune step runs end-to-end on CPU");
        assert_eq!(tier.model_type, "distilbert");
        assert!(tier.trainable_tensors > 0);
    }

    #[test]
    fn target_modules_refusal_fires_through_run_for_bert() {
        let dir = tiny_bert_dir();
        if !dir.join("config.json").exists() {
            eprintln!("skipping: tiny_bert fixture not present at {dir:?}");
            return;
        }
        let params = base_params(dir); // target_modules: None
        let err = run(&params).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--target-modules"));
    }
}
