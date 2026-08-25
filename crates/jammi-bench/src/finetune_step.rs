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
pub(crate) fn triplet_loss(
    a: &Tensor,
    p: &Tensor,
    n: &Tensor,
    margin: f64,
) -> candle_core::Result<Tensor> {
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
}

/// Deterministic synthetic token ids, uniform over `[1, vocab)` so no id is the
/// pad id. An LCG rather than a dependency, and identical across runs so two
/// measurements differ only in the code under test. `pub(crate)`: also the
/// batch generator [`crate::grad_oracle`] drives, so the gradient-oracle and
/// this tier feed an identical synthetic batch for the same `(seed, batch,
/// seq, vocab)` — one LCG, never two copies that could drift.
pub(crate) fn synthetic_ids(
    batch: usize,
    seq: usize,
    vocab: usize,
    seed: u64,
    device: &Device,
) -> Tensor {
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

/// Build the encoder + optimizer + synthetic batch [`run`] drives its step
/// loop against. Factored out of `run()` so a test can build a SEPARATE,
/// independent instance of the identical fixture as a numeric oracle for the
/// update-index-placement test below — never to bypass `run()` (the control
/// flow under test for that fix) as the code path a test drives.
#[allow(clippy::type_complexity)]
fn build_fixture(
    params: &FinetuneStepParams,
) -> Result<
    (
        jammi_encoders::ModernBert,
        candle_nn::AdamW,
        usize,
        Vec<Tensor>,
        Tensor,
    ),
    Box<dyn std::error::Error>,
> {
    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
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
    let trainable_count = trainable.len();
    let opt = candle_nn::AdamW::new(
        trainable,
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

    Ok((encoder, opt, trainable_count, blocks, mask))
}

/// One forward + cosine-margin triplet loss + backward + optimizer step —
/// the exact body [`run`]'s timed loop executes, factored out so the SAME
/// code can also be run once, untimed, before the timed loop starts (see the
/// pre-step call in `run()`), mirroring `torch_finetune_step.py`'s own
/// `_step_once` factoring for the identical reason.
///
/// Returns the step's loss as `f32` — the same value the loop's CUDA-sync
/// read already needed. PLACEMENT: the returned tensor was produced by the
/// forward EARLIER in this call, BEFORE `opt.step(&grads)` a few lines down;
/// reading it after only decides when the host blocks on the (already
/// queued) device work, it does not recompute the loss against the
/// just-updated weights. So the returned value is the PRE-UPDATE loss of
/// THIS call's batch — the weights as they stood when this call STARTED,
/// not as they stand when it returns. `torch_finetune_step.py`'s
/// `_step_once` reads its loss at the mirror-image point for the identical
/// reason, so the two stacks share this placement convention.
fn step_once(
    encoder: &mut jammi_encoders::ModernBert,
    opt: &mut candle_nn::AdamW,
    blocks: &[Tensor],
    mask: &Tensor,
    batch: usize,
    batched_forward: bool,
) -> Result<f32, Box<dyn std::error::Error>> {
    let (a, p, n) = if batched_forward {
        // One forward over the concatenated groups, split after pooling —
        // the trainer's `encode_groups` shape.
        let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0)?;
        let joined_mask = Tensor::cat(&[mask, mask, mask], 0)?;
        let all = encoder.forward(&joined, &joined_mask)?;
        (
            all.narrow(0, 0, batch)?,
            all.narrow(0, batch, batch)?,
            all.narrow(0, 2 * batch, batch)?,
        )
    } else {
        (
            encoder.forward(&blocks[0], mask)?,
            encoder.forward(&blocks[1], mask)?,
            encoder.forward(&blocks[2], mask)?,
        )
    };
    let loss = triplet_loss(&a, &p, &n, 0.3)?;
    let grads = loss.backward()?;
    opt.step(&grads)?;
    // Force completion before returning: candle's CUDA queue is
    // asynchronous, so without this the caller's clock (when called from the
    // timed loop) would measure submission time, not execution time.
    Ok(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?)
}

/// Run the tier and return its report block.
pub fn run(params: &FinetuneStepParams) -> Result<FinetuneStepTier, Box<dyn std::error::Error>> {
    let device_label = match params.cuda_device {
        Some(o) => format!("cuda:{o}"),
        None => "cpu".to_string(),
    };

    let (mut encoder, mut opt, trainable_count, blocks, mask) = build_fixture(params)?;

    // ONE untimed step, BEFORE the timed loop and BEFORE the VRAM/dispatch
    // baselines below — mirrors `torch_finetune_step.py`'s own untimed
    // `_step_once` pre-step (see that file's `run()` for the identical
    // reasoning), so BOTH stacks discard the SAME "update 0" before the
    // officially reported `--warmup` step 0 begins. Without this, jammi's
    // `losses[k]` was the loss after `warmup+k` total optimizer updates
    // while torch's `losses[k]` was the loss after `warmup+k+1` updates (one
    // update ahead) — `loss_first` was the worst case, since
    // `LoraInitMode::ZerosB` makes the LoRA delta identically zero at
    // construction, so jammi's un-fixed `loss_first` was the PRISTINE
    // (zero-optimizer-update) loss while torch's was already one update in.
    // With this pre-step, both stacks' `losses[k]` is the loss after
    // `warmup+k+1` total updates — see
    // `finetune_step_loss_first_is_the_post_pre_step_update` below, which
    // pins this exactly by driving this same `run()` entry point.
    //
    // Never timed, never appended to `losses`/`times`. The VRAM baseline and
    // the dispatch-counter "before" snapshots a few lines down are BOTH
    // taken AFTER this call returns, so this discarded step's own
    // allocation/dispatch activity is absorbed into the baseline / excluded
    // from the delta, never counted as part of the measured run.
    step_once(
        &mut encoder,
        &mut opt,
        &blocks,
        &mask,
        params.batch,
        params.batched_forward,
    )?;

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
    let mut losses = Vec::with_capacity(params.steps);
    for step in 0..(params.warmup + params.steps) {
        let t0 = Instant::now();
        // See `step_once`'s own doc for the loss-value placement convention
        // (PRE-update loss of this call's batch) — unchanged by this
        // refactor, just no longer duplicated inline here.
        let loss_val = step_once(
            &mut encoder,
            &mut opt,
            &blocks,
            &mask,
            params.batch,
            params.batched_forward,
        )?;
        if step >= params.warmup {
            times.push(t0.elapsed().as_secs_f64());
            losses.push(loss_val);
        }
    }

    let ln_dispatch_after = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_after = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_after = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_after = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_after = jammi_lora::lora_epilogue_dispatch_snapshot();
    let lora_linear_fused_dispatch_after = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_after = jammi_encoders::attention_block_dispatch_snapshot();

    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    // `losses` is pushed in lockstep with `times` above (same `if step >=
    // params.warmup` guard, same loop iteration), so it is never empty when
    // `times` is not — the same precondition `times[times.len() / 2]` above
    // already relies on (an `--steps 0` run panics there first).
    let loss_first = *losses.first().expect("losses populated alongside times");
    let loss_last = *losses.last().expect("losses populated alongside times");

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
        trainable_tensors: trainable_count,
        steps_measured: times.len(),
        losses,
        loss_first,
        loss_last,
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

    /// The engine's own tiny 1-layer, 32-hidden ModernBERT fixture — shared
    /// with `jammi-bench`'s `model_inference` tier and `jammi-encoders`'
    /// own tests, referenced (never copied) so this test exercises the SAME
    /// checkpoint format the real GPU path loads. `ModernBertConfig`'s
    /// `serde(default = ...)` fields tolerate the classifier-only keys
    /// (`id2label`/`classifier_pooling`/…) this fixture's `config.json`
    /// also carries — no `deny_unknown_fields` on that struct — and
    /// `ModernBert::builder().build()` only reads the `model.*` backbone
    /// prefix out of the safetensors, ignoring the classifier head weights
    /// this bundle also contains.
    fn tiny_model_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_modernbert_classifier")
    }

    /// A CPU, single-Wqkv-site LoRA config over the tiny fixture — small
    /// enough to run twice in one test with no GPU.
    fn tiny_params() -> FinetuneStepParams {
        FinetuneStepParams {
            model_dir: tiny_model_dir(),
            // 3, deliberately NOT 2: `step_once`'s batched arm computes the
            // negative group's row offset as `2 * b`. At `b == 2`,
            // `2 * b == 2 + b == 4` — a mutation of that `*` to `+` is
            // undetectable by ANY test using `batch: 2` (cargo-mutants
            // caught exactly this: `replace * with + in step_once`
            // survived until this fixture moved off `b == 2` — see
            // `crates/jammi-bench/src/grad_oracle.rs`'s identical fixture
            // note for the sibling tier where this was first caught).
            batch: 3,
            seq: 8,
            steps: 2,
            warmup: 1,
            lora_rank: 1,
            lora_alpha: 1.0,
            lora_dropout: 0.0,
            target_modules: vec!["Wqkv".to_string()],
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            cuda_device: None,
            seed: 1,
            batched_forward: true,
        }
    }

    /// The `*_fused_dispatches` / `*_eager_dispatches` counter fields on
    /// [`FinetuneStepTier`] MUST be a DELTA over the process-wide dispatch
    /// registries — `run()`'s `*_dispatch_before` snapshot taken right
    /// before the step loop, subtracted (via `saturating_sub`) from the
    /// `*_dispatch_after` snapshot taken right after it — never the raw
    /// ambient "after" total. The registries are process-global counters
    /// (`jammi_kernels::admission::DispatchCounters`, atomics shared by
    /// every call site in the process, including an EARLIER call to this
    /// same `run()`), so a report that skipped the "before" subtraction
    /// would leak an earlier run's (or an earlier test's) dispatch activity
    /// into this run's numbers.
    ///
    /// Proven directly, not inferred: run the identical tiny config through
    /// `run()` TWICE in one process and assert the two reports carry the
    /// SAME per-counter totals. If `run()` reported the raw "after"
    /// snapshot instead of `after - before`, the second call's totals would
    /// include the first call's dispatches too and come out roughly double
    /// — this test reddens the moment that subtraction is removed (the
    /// acceptance gap this test closes: deleting the before/after snapshot
    /// pair and reporting the bare "after" counters previously had no test
    /// noticing).
    #[test]
    fn finetune_step_counters_are_a_snapshot_delta_not_a_running_total() {
        // `cargo test` runs this crate's tests on multiple threads by
        // default, and `model_inference`'s own tests build and forward
        // real encoders in the same process — so the process-global
        // dispatch registries these counters read are NOT exclusive to
        // this test. An exact-equality check between two back-to-back
        // `run()` calls is therefore genuinely flaky (observed: a 6-count
        // gap from concurrent noise on an otherwise-correct implementation)
        // and would not distinguish "noisy" from "actually broken". So
        // instead of comparing two like-sized runs for exact equality, this
        // DELIBERATELY manufactures a large, known step of ambient-counter
        // growth (the "inflation" run, ~50x a baseline run's own step
        // count) BETWEEN two baseline-sized runs, and asserts the second
        // baseline-sized run's totals stayed close to the first's — a
        // generous absolute tolerance absorbs ordinary cross-test noise
        // (single digits, per the observation above) without absorbing the
        // inflation run's ~50x contribution, which a raw (non-delta)
        // counter would leak straight into the report.
        let small = tiny_params();
        let mut big = small.clone();
        big.steps = 100;
        big.warmup = 0;

        let baseline = run(&small).expect("baseline finetune-step run");
        let _inflate = run(&big).expect("inflation finetune-step run (report discarded)");
        let after_inflation = run(&small).expect("post-inflation finetune-step run");

        // Encoder-side counters: every training step's forward touches
        // LayerNorm (embeddings + MLP norm), RoPE, the masked softmax, and
        // GeGLU at least once regardless of which linear carries a LoRA
        // adapter, so each pair's total is non-zero.
        let ln_total = |t: &FinetuneStepTier| t.ln_fused_dispatches + t.ln_eager_dispatches;
        let rope_total = |t: &FinetuneStepTier| t.rope_fused_dispatches + t.rope_eager_dispatches;
        let softmax_total =
            |t: &FinetuneStepTier| t.softmax_fused_dispatches + t.softmax_eager_dispatches;
        let geglu_total =
            |t: &FinetuneStepTier| t.geglu_fused_dispatches + t.geglu_eager_dispatches;
        let attention_block_total = |t: &FinetuneStepTier| {
            t.attention_block_fused_dispatches + t.attention_block_eager_dispatches
        };

        for (name, total_of) in [
            ("ln", ln_total as fn(&FinetuneStepTier) -> u64),
            ("rope", rope_total),
            ("softmax", softmax_total),
            ("geglu", geglu_total),
            ("attention_block", attention_block_total),
        ] {
            let base = total_of(&baseline);
            let after = total_of(&after_inflation);
            assert!(base > 0, "{name}: expected at least one dispatch per run");
            let tolerance = base.max(4) * 4 + 40;
            assert!(
                after <= base + tolerance,
                "{name}: post-inflation total {after} vs baseline {base} \
                 (tolerance {tolerance}) — the 100-step inflation run's dispatches \
                 appear to have leaked into this report; counters must be a \
                 before/after snapshot DELTA over this run's own window, not the \
                 raw ambient (process-lifetime) total"
            );
        }

        // LoRA-side counters: `target_modules = ["Wqkv"]` guarantees exactly
        // one adapted linear, but WHICH counter family carries the
        // dispatches (the standalone epilogue vs the fused whole-site
        // kernel) is an admission-predicate outcome, not something this
        // test pins — see `lora_linear_fused_dispatches`'s own doc for why
        // the epilogue pair is permanently zero on a run where the fused
        // LoRA-site counter is non-zero. Sum both families and apply the
        // same inflation-leak check to the sum.
        let lora_total = |t: &FinetuneStepTier| {
            t.lora_epilogue_fused_dispatches
                + t.lora_epilogue_eager_dispatches
                + t.lora_linear_fused_dispatches
                + t.lora_linear_eager_dispatches
        };
        let (lora_base, lora_after) = (lora_total(&baseline), lora_total(&after_inflation));
        assert!(
            lora_base > 0,
            "expected the LoRA-adapted Wqkv site to dispatch at least once"
        );
        let lora_tolerance = lora_base.max(4) * 4 + 40;
        assert!(
            lora_after <= lora_base + lora_tolerance,
            "lora: post-inflation total {lora_after} vs baseline {lora_base} \
             (tolerance {lora_tolerance}) — the inflation run's LoRA-site \
             dispatches appear to have leaked into this report; counters must be \
             a before/after snapshot DELTA, not the raw ambient total"
        );
    }

    /// `losses`/`loss_first`/`loss_last` are populated from the SAME
    /// post-`opt.step` read the step loop already does for its CUDA sync
    /// (see the loss-read call site's own comment) — one entry per measured
    /// step, warmup excluded, `loss_first`/`loss_last` matching the ends of
    /// `losses`. Every value must be finite: a NaN/Inf loss silently
    /// poisoning `losses` while `loss_first`/`loss_last` still "looked like
    /// numbers" would be exactly the non-finite-control gap the fixtures
    /// contract warns about (`NaN > c` is `false`), so this asserts
    /// `is_finite()` explicitly rather than only checking shape.
    #[test]
    fn finetune_step_losses_track_measured_steps_and_are_finite() {
        let params = tiny_params();
        let tier = run(&params).expect("finetune-step run");

        assert_eq!(tier.losses.len(), params.steps);
        assert_eq!(tier.losses.len(), tier.steps_measured);
        assert_eq!(tier.loss_first, tier.losses[0]);
        assert_eq!(tier.loss_last, tier.losses[tier.losses.len() - 1]);

        // Theoretical range of `triplet_loss` (margin=0.3, cosines in
        // [-1, 1]): `relu(margin - cos(a,p) + cos(a,n))` is in
        // `[0, margin + 2]`, and `mean_all` over that range stays in it —
        // so every recorded value must land in `[0, margin + 2]` with a
        // small float-slop margin. A finiteness check alone would NOT
        // catch a mutation that hardcodes the read to a fixed in-range
        // constant (e.g. always `0.0`); the bound plus the non-degenerate
        // check below together do.
        const MAX_TRIPLET_LOSS: f32 = 0.3 + 2.0 + 1e-3;
        for (i, &l) in tier.losses.iter().enumerate() {
            assert!(l.is_finite(), "losses[{i}] = {l} is not finite");
            assert!(
                (0.0..=MAX_TRIPLET_LOSS).contains(&l),
                "losses[{i}] = {l} is outside triplet_loss's theoretical range [0, {MAX_TRIPLET_LOSS}]"
            );
        }
        // Non-degenerate: a mutation that hardcoded the loss read to a
        // fixed constant (in-range, e.g. `0.0`) would still pass every
        // check above. A real triplet-margin loss over a random-init tiny
        // model's random synthetic tokens landing at EXACTLY the same
        // float value on every one of the measured steps is not something
        // real floating-point computation does (LoRA weights move every
        // step via AdamW, even against the same fixed batch) — so requiring
        // at least one value to differ from the first is a cheap, reliable
        // non-degeneracy check without needing a numeric oracle.
        assert!(
            tier.losses.iter().any(|&l| l != tier.losses[0]),
            "every measured loss is bit-identical ({:?}) — looks like the loss read was \
             replaced by a fixed constant instead of the real per-step tensor value",
            tier.losses
        );
    }

    /// B1 fix pin: `run()` must execute exactly ONE untimed optimizer update
    /// (the pre-step) BEFORE its timed loop starts recording, so
    /// `tier.losses[0]` is the loss after `warmup+1` total optimizer
    /// updates, never the PRISTINE (zero-update) loss. `LoraInitMode::ZerosB`
    /// makes this distinguishable: `B` is zero-initialized, so the LoRA
    /// delta — and therefore the forward's output — does not depend on `A`'s
    /// random draw at all until the first update moves `B` off zero; the
    /// pristine and post-one-update losses are computed from a materially
    /// different forward, not just numerically close.
    ///
    /// Drives the REAL `run()` entry point (clause 8: never `step_once` in
    /// isolation as the code path under test) and compares its `losses[0]`
    /// against an INDEPENDENT oracle: a SEPARATE fixture built via the same
    /// `build_fixture` construction `run()` itself uses, then stepped by
    /// hand through the exact production `step_once` primitive `run()`
    /// itself calls. This pins the CONTROL-FLOW claim — "`run()` calls
    /// `step_once` once, untimed, before recording starts" — via an
    /// independently reconstructed trajectory, rather than re-deriving the
    /// loss arithmetic from scratch (which `step_once`'s own non-degeneracy
    /// test above already covers).
    #[test]
    fn finetune_step_loss_first_is_the_post_pre_step_update() {
        let params = FinetuneStepParams {
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };

        // Independent oracle: a SEPARATE fixture from the identical params
        // (same seed => same synthetic ids, same LoRA init draw — jammi's
        // seeded init is a pure function of (seed, parameter name),
        // independent of construction order, so two independent
        // `build_fixture` calls with the same `params` start bit-identical).
        // `lora_dropout: 0.0` in `tiny_params()` means no dropout RNG is in
        // play either, so this replay is exactly reproducible on CPU/F32.
        let (mut o_encoder, mut o_opt, _count, o_blocks, o_mask) =
            build_fixture(&params).expect("oracle fixture");
        // Call #1: the PRISTINE (zero prior updates) loss.
        let pristine_loss = step_once(
            &mut o_encoder,
            &mut o_opt,
            &o_blocks,
            &o_mask,
            params.batch,
            params.batched_forward,
        )
        .expect("oracle pre-step");
        // Call #2: PRE-update loss with exactly ONE prior update applied —
        // this is what `run()` is claimed to report as `losses[0]` once its
        // own untimed pre-step (call #1's counterpart) has run.
        let expected_loss_first = step_once(
            &mut o_encoder,
            &mut o_opt,
            &o_blocks,
            &o_mask,
            params.batch,
            params.batched_forward,
        )
        .expect("oracle second step (this call's PRE-update loss is losses[0])");

        let tier = run(&params).expect("finetune-step run");

        assert_eq!(tier.losses.len(), 1);
        assert_eq!(
            tier.losses[0], expected_loss_first,
            "losses[0] must equal the PRE-update loss of the SECOND optimizer \
             update (one untimed pre-step, then the first recorded step) — \
             run() must call step_once exactly once, untimed, before its \
             timed loop begins recording, mirroring torch_finetune_step.py's \
             own untimed _step_once pre-step so the two stacks' loss \
             trajectories line up on the same absolute update index"
        );
        // Explicitly distinguish from the PRISTINE (zero-update) loss,
        // pinning that the recorded value is NOT the pre-B1-fix placement
        // (loss_first used to be exactly this value).
        assert_ne!(
            tier.losses[0], pristine_loss,
            "losses[0] equals the PRISTINE (zero-optimizer-update) loss — \
             looks like the B1 regression re-appeared: run() is no longer \
             executing its untimed pre-step before recording starts"
        );
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `build_fixture`'s
    /// `params.seed + i` block-offset arithmetic surviving as `seed * i`):
    /// at `seed == 0`, the CORRECT `seed + i` gives three DISTINCT synthetic
    /// seeds (0, 1, 2) for anchor/positive/negative, while the mutant
    /// `seed * i` collapses ALL THREE to seed `0` (`0*0 = 0*1 = 0*2 = 0`),
    /// making the three groups literally IDENTICAL token ids. Since the
    /// encoder is a deterministic function of its input, identical inputs
    /// force identical embeddings, which forces `cos(a,p) == cos(a,n)` and
    /// pins `triplet_loss` at EXACTLY `margin` (`0.3`) — a recognizable,
    /// cheap-to-check signature that does not require exposing token ids
    /// from `FinetuneStepTier`'s own (separately pinned-key-set) schema at
    /// all. Blocks are built ONCE before the step loop and reused every
    /// iteration, so this degeneracy — if the mutant were live — would
    /// hold for every measured step, not just the first.
    #[test]
    fn finetune_step_synthetic_batch_offset_is_seed_plus_group_index_not_seed_times_group_index() {
        let mut params = tiny_params();
        params.seed = 0;
        let tier = run(&params).expect("finetune-step run");
        for (i, &loss) in tier.losses.iter().enumerate() {
            assert!(
                (loss - 0.3).abs() > 1e-4,
                "losses[{i}] = {loss} landed suspiciously close to margin (0.3) at seed=0 — \
                 looks like the three synthetic groups collapsed to IDENTICAL token ids \
                 (consistent with `seed + i` having become `seed * i`, which is 0 for every \
                 group at seed=0)"
            );
        }
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `step_once`'s batched
    /// arm's `all.narrow(0, 2 * batch, batch)` — the negative group's row
    /// offset — surviving as `2 + batch`/`2 / batch`, even after
    /// `tiny_params()` moved off the degenerate `batch == 2` case: at
    /// `batch == 3` a wrong offset picks an OVERLAPPING-but-different
    /// slice, not an out-of-bounds or obviously-degenerate one, so
    /// `finetune_step_synthetic_batch_offset_is_seed_plus_group_index_not_seed_times_group_index`'s
    /// margin-collapse signature does not fire either).
    ///
    /// `jammi_lora`'s seeded init is a pure function of `(seed, parameter
    /// name)`, and `synthetic_ids` is a pure function of `(seed, batch,
    /// seq, vocab)` — so two `run()` calls with the SAME `seed` and
    /// EVERYTHING else equal except `batched_forward` start from identical
    /// weights and feed identical per-group tokens, differing only in
    /// whether the three groups go through ONE joined forward (`narrow`-
    /// split) or three separate ones (`blocks[0]`/`blocks[1]`/`blocks[2]`
    /// directly, no arithmetic at all). ModernBERT's per-row attention
    /// mask means no row can see another row's tokens, so the two arms
    /// must produce the SAME losses — a miscomputed row offset in the
    /// batched arm silently picks the WRONG rows there while the unbatched
    /// arm stays correct, and the two diverge exactly when this arithmetic
    /// is wrong.
    #[test]
    fn finetune_step_batched_and_unbatched_forward_agree() {
        let mut batched_params = tiny_params();
        batched_params.batched_forward = true;
        let batched = run(&batched_params).expect("batched run");

        let mut unbatched_params = tiny_params();
        unbatched_params.batched_forward = false;
        let unbatched = run(&unbatched_params).expect("unbatched run");

        assert_eq!(batched.losses.len(), unbatched.losses.len());
        // NOT bit-exact -- see `crates/jammi-bench/src/grad_oracle.rs`'s
        // sibling test for why (measured: candle's batched, 3*batch-row
        // matmul is free to reduce in a different order than three
        // separate batch-row matmuls -- mathematically equivalent, not
        // bitwise so, since f32 addition is not associative). `TOL_REL`/
        // `TOL_ABS` are generous relative to that measured noise floor
        // while staying far tighter than a real group-selection defect
        // would clear.
        const TOL_REL: f32 = 1e-3;
        const TOL_ABS: f32 = 1e-6;
        for (i, (&x, &y)) in batched
            .losses
            .iter()
            .zip(unbatched.losses.iter())
            .enumerate()
        {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs());
            assert!(
                diff <= TOL_ABS + TOL_REL * scale,
                "losses[{i}]: batched={x} vs unbatched={y} (|diff|={diff}) exceeds the \
                 floating-point reduction-order noise tolerance (abs {TOL_ABS} + rel \
                 {TOL_REL}*{scale}) -- looks like a real divergence (e.g. narrow(0, 2*batch, \
                 batch) miscomputed, picking the WRONG rows in the batched arm), not rounding \
                 noise"
            );
        }
    }
}
