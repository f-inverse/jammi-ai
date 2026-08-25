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
    /// The op key set the CALLER intended `JAMMI_KERNELS_DISABLE` to carry
    /// into this process, sorted — `None` when the caller makes no claim
    /// (the ordinary, unchecked case). When `Some`, [`run`] hard-errors if
    /// [`jammi_kernels::admission::disabled_ops_requested`] does not read
    /// back EXACTLY this set, turning the "env var silently dropped"
    /// failure mode (a var-NAME typo, an unforwarded ssh/`docker -e`
    /// environment — see `kernels_disabled_requested`/`kernels_disabled_fired`'s
    /// doc just above [`run`]'s dispatch-counter reads) into a machine-
    /// enforced check on the SAME invocation that intended the disable,
    /// rather than something a caller has to eyeball in the emitted JSON
    /// report after the fact. Distinct from `unmatched_disables()` (which
    /// this function already checks): that catches a REQUESTED entry that
    /// never fired; this catches the process never having received the
    /// request it was TOLD to expect at all, at any point.
    pub expect_kernels_disabled: Option<Vec<String>>,
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
///
/// Returns `Err` — not a `FinetuneStepTier` with a suspiciously-clean
/// dispatch split — when `JAMMI_KERNELS_DISABLE` named an op key that
/// never actually disabled a live dispatch this run (see the check just
/// before this function returns, and
/// `jammi_kernels::admission::unmatched_disables`'s doc): that is a typo
/// in the disable list, not evidence the forced-eager arm ran, and must
/// never be reported as a datum (contract K-aux). Also returns `Err` when
/// `params.expect_kernels_disabled` is `Some` and does not match what this
/// process's `JAMMI_KERNELS_DISABLE` actually resolved to — see
/// `FinetuneStepParams::expect_kernels_disabled`'s doc.
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
    // Same mechanism, for the P2 fused LoRA SITE (base matmul + dropout +
    // both LoRA GEMMs + epilogue, one CustomOp3) — see
    // `jammi_lora::lora_linear_fused_dispatch_snapshot`'s doc for why
    // `lora_epilogue_*` above legitimately reads zero on a run where this
    // one is nonzero.
    let lora_linear_fused_dispatch_before = jammi_lora::lora_linear_fused_dispatch_snapshot();
    // Same mechanism, for the fused whole-attention-block kernel.
    let attention_block_dispatch_before = jammi_encoders::attention_block_dispatch_snapshot();

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
    let lora_linear_fused_dispatch_after = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_after = jammi_encoders::attention_block_dispatch_snapshot();

    // `JAMMI_KERNELS_DISABLE` safety property (contract K-aux): a
    // disable-list entry that never actually disabled a live `admit` call
    // this run is a typo, not a forced-eager measurement — this run is
    // INVALID, not a datum with a plausible-looking JSON tier. Checked at
    // the END of the run, after every dispatch site above has had its
    // chance to fire: `jammi_kernels::admission`'s registry is populated
    // by observation and cannot be validated at process start (see
    // `jammi_kernels::admission::unmatched_disables`'s doc).
    let unmatched_kernel_disables = jammi_kernels::admission::unmatched_disables();
    if !unmatched_kernel_disables.is_empty() {
        return Err(format!(
            "JAMMI_KERNELS_DISABLE named op key(s) that never disabled a live \
             dispatch this run (INVALID run, not a datum): {unmatched_kernel_disables:?}"
        )
        .into());
    }

    // The RESOLVED disable state, for the report artifact (contract K-aux,
    // round 2 / B3): `requested` is what `JAMMI_KERNELS_DISABLE` named this
    // process (sorted, empty when unset); `fired` is which of those entries
    // actually disabled a live dispatch this run (a strict subset once
    // `unmatched_kernel_disables` is empty, per the check just above).
    // Recorded even on an ordinary undisabled run (both empty), rather than
    // only on a forced-eager one: an omitted pair would read as "this
    // report predates the field", which is false — every report from this
    // build carries an opinion on which arm it measured. This is what lets
    // a downstream A/B harness catch the "env var silently dropped" failure
    // mode `unmatched_disables` does NOT cover: a run whose
    // `JAMMI_KERNELS_DISABLE` never reached this process at all (a var-NAME
    // typo, an unforwarded ssh/`docker -e` environment) has NOTHING
    // requested, so `unmatched_disables()` is trivially empty and the run
    // succeeds — but `kernels_disabled_requested`/`kernels_disabled_fired`
    // then both read `[]` here too, distinguishing it from a genuine
    // forced-eager run (both non-empty and equal) that a caller intended to
    // compare against.
    let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
    let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();

    // `--expect-kernels-disabled` (contract K-aux, round 2 advisory): the
    // binary controls its own argv, so a caller that names its intended
    // `JAMMI_KERNELS_DISABLE` set on the SAME command line gets a hard
    // error instead of having to notice, after the fact, that the
    // process-observed `kernels_disabled_requested` in the emitted JSON
    // report came back empty (or different) — machine-enforced rather than
    // eyeballed. Compared as SORTED sets: `kernels_disabled_requested` is
    // already sorted, and the expectation is sorted here so the caller's
    // ordering on the command line is not load-bearing.
    if let Some(expected) = &params.expect_kernels_disabled {
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();
        if kernels_disabled_requested != expected_sorted {
            return Err(format!(
                "--expect-kernels-disabled named {expected_sorted:?} but this process's real \
                 JAMMI_KERNELS_DISABLE resolved to {kernels_disabled_requested:?} — the env var \
                 was dropped, mistyped, or not forwarded to this process (INVALID run, not a \
                 datum)"
            )
            .into());
        }
    }

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
        lora_linear_fused_dispatches: lora_linear_fused_dispatch_after
            .fused
            .saturating_sub(lora_linear_fused_dispatch_before.fused),
        lora_linear_eager_dispatches: lora_linear_fused_dispatch_after
            .eager
            .saturating_sub(lora_linear_fused_dispatch_before.eager),
        attention_block_fused_dispatches: attention_block_dispatch_after
            .fused
            .saturating_sub(attention_block_dispatch_before.fused),
        attention_block_eager_dispatches: attention_block_dispatch_after
            .eager
            .saturating_sub(attention_block_dispatch_before.eager),
        kernels_disabled_requested,
        kernels_disabled_fired,
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
