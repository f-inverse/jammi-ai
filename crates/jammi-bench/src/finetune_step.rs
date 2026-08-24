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

use jammi_ai::fine_tune::optimizer::clip_gradients;

use crate::report::{FinetuneStepTier, Measurement};

/// A `--max-grad-norm` value the caller explicitly supplied but that would
/// either silently disable clipping ([`clip_gradients`]'s own `<= 0.0`
/// convention, which reads a "clip on" row as clipped when it silently was
/// not) or propagate a non-finite coefficient into every gradient. A step
/// this contract measures must be the step it claims to measure, so a bad
/// explicit value is refused rather than folded into the "absent" no-op
/// path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidMaxGradNorm(pub f32);

impl std::fmt::Display for InvalidMaxGradNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "--max-grad-norm {} is invalid: must be finite and > 0.0 \
             (a non-finite or <= 0.0 value would either corrupt every \
             gradient or silently disable clipping, mislabeling this row)",
            self.0
        )
    }
}

impl std::error::Error for InvalidMaxGradNorm {}

/// Refuse a non-finite or non-positive `--max-grad-norm`. `None` (flag
/// absent) is never passed here — this only guards a value the operator
/// explicitly supplied.
fn validate_max_grad_norm(v: f32) -> Result<f32, InvalidMaxGradNorm> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(InvalidMaxGradNorm(v))
    }
}

/// How many times this run's step loop invoked the production
/// [`clip_gradients`] — the counted fact backing the "clip on" A/B row,
/// rather than a log line an operator has to trust. Process-wide, so tests
/// read it as a before/after delta the same way the fused-kernel dispatch
/// counters above are read.
static CLIP_INVOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Snapshot the clip-invocation counter. Test-only: the counted-fact channel
/// the invocation-count tests below read as a before/after delta; nothing in
/// the production report reads it today.
#[cfg(test)]
fn clip_invocations_snapshot() -> u64 {
    CLIP_INVOCATIONS.load(Ordering::Relaxed)
}

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
    pub target_modules: Vec<String>,
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
    /// `Some(max_norm)` runs the PRODUCTION [`clip_gradients`] after backward
    /// and before the optimizer step, at the same point in the sequence the
    /// trainer does (`ctx.optimizer.set_learning_rate` →
    /// [`jammi_ai::fine_tune::optimizer::clip_and_step`], which is
    /// `clip_gradients` then `optimizer.step`). `None` (the default) skips
    /// clipping entirely and is bit-identical to this tier's behaviour before
    /// this field existed.
    ///
    /// This exists because the shipped trainer's default config
    /// (`max_grad_norm = 1.0`) always clips: skipping it, as this tier did
    /// before, measures a step the product never runs — `clip_gradients`
    /// does one `to_scalar` D2H sync per trainable `Var` (224 on a
    /// ModernBERT-large r16 `Wqkv`/`Wo`/`Wi` LoRA config), a host-sync cost
    /// no `None` row can see. Recording both an on and an off row on the
    /// same box makes that cost a measured delta instead of an assumption.
    pub max_grad_norm: Option<f32>,
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
    if let Some(max_norm) = params.max_grad_norm {
        validate_max_grad_norm(max_norm)?;
    }

    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
    };
    let device_label = match params.cuda_device {
        Some(o) => format!("cuda:{o}"),
        None => "cpu".to_string(),
    };

    let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))?;
    let config: jammi_encoders::ModernBertConfig = serde_json::from_str(&config_raw)?;
    let weights = params.model_dir.join("model.safetensors");

    let varmap = VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &params.target_modules,
        layers_to_transform: &None,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        use_rslora: false,
        lora_dropout: (params.lora_dropout > 0.0).then_some(params.lora_dropout),
        rank_pattern: &empty_ranks,
        init_mode: jammi_lora::LoraInitMode::ZerosB,
        seed: params.seed,
    };

    let mut encoder = jammi_encoders::ModernBert::builder()
        .pooling(jammi_encoders::Pooling::Mean)
        .backbone_dtype(jammi_encoders::compute_precision_to_dtype(
            params.backbone_dtype,
        ))
        .lora(lora)
        .build(&[weights.as_path()], &config, &device, &varmap)?;
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
                config.vocab_size,
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
        let mut grads = loss.backward()?;
        // Same point in the sequence the trainer clips at: after backward,
        // before the optimizer step (trainer.rs's `process_batch_loss` runs
        // `scaled_loss.backward()` then, at the accumulation boundary,
        // `clip_and_step` — clip_gradients then `optimizer.step` — never the
        // reverse). `None` skips this block entirely: bit-identical to the
        // step this tier measured before `max_grad_norm` existed.
        if let Some(max_norm) = params.max_grad_norm {
            clip_gradients(&trainable, &mut grads, max_norm as f64)
                .map_err(|e| format!("finetune-step clip_gradients: {e}"))?;
            CLIP_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
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
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_dropout: params.lora_dropout as f64,
        target_modules: params.target_modules.clone(),
        batched_forward: params.batched_forward,
        max_grad_norm: params.max_grad_norm,
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
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// `CLIP_INVOCATIONS` is one process-wide counter; these tests read it as
    /// a before/after delta around their own `run` call, which is only valid
    /// if no other test's `run` call can interleave and add its own
    /// invocations inside that window. Cargo runs `#[test]` fns on multiple
    /// threads by default, so every test below that touches the counter
    /// holds this mutex for its full body.
    static CLIP_COUNTER_SERIAL: Mutex<()> = Mutex::new(());

    /// The committed, generic ModernBERT fixture
    /// (`tests/fixtures/tiny_modernbert/config.json` +
    /// `model.safetensors`) — synthetic, tiny (hidden=32, 1 layer),
    /// no consumer data shape. Located the same way
    /// `jammi_test_utils::workspace_root` does (two levels up from
    /// `CARGO_MANIFEST_DIR`); `jammi-test-utils` is not a dev-dependency of
    /// this crate today and adding one is a `Cargo.toml` edit outside this
    /// change's declared file scope, so the two-line walk is duplicated here
    /// rather than diverging from it.
    fn tiny_modernbert_dir() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<name>")
            .parent()
            .expect("workspace root")
            .join("tests")
            .join("fixtures")
            .join("tiny_modernbert")
    }

    fn tiny_target_modules() -> Vec<String> {
        vec!["Wqkv".to_string(), "Wo".to_string(), "Wi".to_string()]
    }

    fn tiny_params(max_grad_norm: Option<f32>, steps: usize, warmup: usize) -> FinetuneStepParams {
        FinetuneStepParams {
            model_dir: tiny_modernbert_dir(),
            batch: 2,
            seq: 4,
            steps,
            warmup,
            lora_rank: 2,
            lora_alpha: 4.0,
            lora_dropout: 0.0,
            target_modules: tiny_target_modules(),
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            cuda_device: None,
            seed: 42,
            batched_forward: true,
            max_grad_norm,
        }
    }

    #[test]
    fn clip_gradients_invocation_count_equals_measured_steps() {
        let _serial = CLIP_COUNTER_SERIAL.lock().unwrap();
        let before = clip_invocations_snapshot();
        let params = tiny_params(Some(1.0), 3, 0);
        let tier = run(&params).expect("finetune-step run with --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(tier.steps_measured, 3);
        assert_eq!(
            after - before,
            3,
            "clip_gradients must be invoked exactly once per step, not logged \
             and trusted"
        );
    }

    #[test]
    fn clip_gradients_invocation_count_includes_warmup_iterations() {
        let _serial = CLIP_COUNTER_SERIAL.lock().unwrap();
        let before = clip_invocations_snapshot();
        // warmup=1, steps=2: the trainer clips every step it runs, warmup
        // included (the loop that clips is the same loop that warms up), so
        // the counted fact must reflect all 3 loop iterations, not just the
        // 2 that land in `steps_measured`.
        let params = tiny_params(Some(1.0), 2, 1);
        let tier = run(&params).expect("finetune-step run with --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(tier.steps_measured, 2);
        assert_eq!(after - before, 3, "clip must run during warmup too");
    }

    #[test]
    fn clip_gradients_never_invoked_when_max_grad_norm_absent() {
        let _serial = CLIP_COUNTER_SERIAL.lock().unwrap();
        let before = clip_invocations_snapshot();
        let params = tiny_params(None, 2, 0);
        run(&params).expect("finetune-step run without --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(
            after, before,
            "clip_gradients must not run at all when --max-grad-norm is absent"
        );
    }

    #[test]
    fn max_grad_norm_rejects_zero() {
        let params = tiny_params(Some(0.0), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=0.0 must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_negative() {
        let params = tiny_params(Some(-1.0), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=-1.0 must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_nan() {
        let params = tiny_params(Some(f32::NAN), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=NaN must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_infinite() {
        let params = tiny_params(Some(f32::INFINITY), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=inf must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    /// Runs the same construction `run` does (model + LoRA + AdamW, two
    /// forward/backward/(clip)/step iterations on CPU) and returns the final
    /// flattened trainable-parameter values. `run`'s own report cannot carry
    /// this signal — a step time is never a proxy for a parameter's bits —
    /// so the determinism tests below reconstruct the harness directly
    /// rather than reading it off [`FinetuneStepTier`].
    fn train_two_steps_and_flatten_params(max_grad_norm: Option<f32>) -> Vec<f32> {
        let device = Device::Cpu;
        let dir = tiny_modernbert_dir();
        let config_raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let config: jammi_encoders::ModernBertConfig =
            serde_json::from_str(&config_raw).expect("parse config.json");
        let weights = dir.join("model.safetensors");

        let varmap = VarMap::new();
        let empty_ranks = std::collections::HashMap::new();
        let target_modules = tiny_target_modules();
        let lora = jammi_lora::LoraBuildConfig {
            target_modules: &target_modules,
            layers_to_transform: &None,
            lora_rank: 2,
            lora_alpha: 4.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &empty_ranks,
            init_mode: jammi_lora::LoraInitMode::ZerosB,
            seed: 42,
        };
        let mut encoder = jammi_encoders::ModernBert::builder()
            .pooling(jammi_encoders::Pooling::Mean)
            .backbone_dtype(DType::F32)
            .lora(lora)
            .build(&[weights.as_path()], &config, &device, &varmap)
            .expect("build encoder");
        encoder.set_training(true);

        let trainable = varmap.all_vars();
        assert!(!trainable.is_empty(), "target_modules matched nothing");
        let mut opt = candle_nn::AdamW::new(
            trainable.clone(),
            candle_nn::ParamsAdamW {
                lr: 2e-4,
                weight_decay: 0.01,
                ..Default::default()
            },
        )
        .expect("build optimizer");

        let mask = Tensor::ones((2, 4), DType::U32, &device).expect("mask");
        let blocks: Vec<Tensor> = (0..3)
            .map(|i| synthetic_ids(2, 4, config.vocab_size, 42 + i, &device))
            .collect();

        for _ in 0..2 {
            let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0).expect("cat ids");
            let joined_mask = Tensor::cat(&[&mask, &mask, &mask], 0).expect("cat mask");
            let all = encoder.forward(&joined, &joined_mask).expect("forward");
            let a = all.narrow(0, 0, 2).expect("narrow a");
            let p = all.narrow(0, 2, 2).expect("narrow p");
            let n = all.narrow(0, 4, 2).expect("narrow n");
            let loss = triplet_loss(&a, &p, &n, 0.3).expect("triplet loss");
            let mut grads = loss.backward().expect("backward");
            if let Some(max_norm) = max_grad_norm {
                clip_gradients(&trainable, &mut grads, max_norm as f64).expect("clip_gradients");
            }
            opt.step(&grads).expect("optimizer step");
        }

        sorted_params_snapshot(&varmap)
    }

    /// Every trainable parameter's flattened values, iterated in a
    /// deterministic (sorted-by-name) order.
    ///
    /// `VarMap::all_vars()` returns its `HashMap<String, Var>`'s iteration
    /// order, which is NOT stable across separate `VarMap` instances — std's
    /// `HashMap` reseeds its hasher per construction, so two freshly-built
    /// `VarMap`s from the identical seed/config produce their vars in
    /// different orders (confirmed empirically: an earlier draft of this
    /// test compared `all_vars()`'s raw order directly and false-failed on
    /// every second run — the underlying per-parameter values were bit-
    /// identical, just concatenated in a different sequence). Reading via
    /// [`candle_nn::VarMap::data`] and sorting by name removes that
    /// ordering as a variable, so a mismatch here is a real trajectory
    /// difference, not a HashMap artifact.
    fn sorted_params_snapshot(varmap: &VarMap) -> Vec<f32> {
        let data = varmap.data().lock().expect("VarMap data mutex");
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        names
            .into_iter()
            .flat_map(|name| {
                let t: &Tensor = &data[name];
                t.flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("to_vec1")
            })
            .collect()
    }

    /// Absent `--max-grad-norm`: two runs from the same seed take the same
    /// trajectory bit-for-bit. This is the oracle for "bit-identical to
    /// before this commit" — the absent path never calls `clip_gradients`
    /// (see `clip_gradients_never_invoked_when_max_grad_norm_absent` above),
    /// so its op sequence is unchanged by this commit; this test pins that
    /// the resulting parameters are actually deterministic, not merely that
    /// the code looks unchanged on inspection.
    #[test]
    fn absent_max_grad_norm_is_deterministic_across_runs() {
        let a = train_two_steps_and_flatten_params(None);
        let b = train_two_steps_and_flatten_params(None);
        assert_eq!(
            a, b,
            "same seed, same steps, no clipping: parameters must be \
             bit-identical run-to-run"
        );
    }

    /// A huge `max_norm` (1e9) matches the absent path bit-for-bit on the
    /// HOST implementation: `clip_gradients` computes `total_norm` and,
    /// finding it far below `max_norm`, takes the early `total_norm <=
    /// max_norm` return — no gradient is ever touched, not even multiplied
    /// by a coefficient of 1.0. This identity holds for THIS (host) path
    /// only; P4b(ii)'s device-side clip may reduce differently (e.g. an
    /// unconditional `broadcast_mul` per the torch-order convention this
    /// contract's Facts section cites) and will need its own re-pin when it
    /// lands.
    #[test]
    fn huge_max_grad_norm_matches_absent_on_host() {
        let absent = train_two_steps_and_flatten_params(None);
        let huge = train_two_steps_and_flatten_params(Some(1e9));
        assert_eq!(
            absent, huge,
            "clip_gradients's early return (total_norm <= max_norm) must \
             leave every gradient untouched when max_norm is far above the \
             actual grad norm"
        );
    }
}
