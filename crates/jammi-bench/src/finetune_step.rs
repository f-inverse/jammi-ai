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

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// The Jammi-owned, fused-kernel-wired AdamW (`jammi_ai::fine_tune::adamw::AdamW`),
// not `candle_nn::AdamW`: this tier measures the step the shipped trainer
// actually runs (`fine_tune::trainer::TrainingLoop` builds its optimizer via
// THIS type — see that module), and the fused/eager dispatch split
// (`adamw_fused_dispatches`/`adamw_eager_dispatches` on `FinetuneStepTier`)
// is only a real, non-vacuous signal if the step loop dispatches through the
// SAME `admit`-gated path production does, not a foreign optimizer that
// never touches `jammi_kernels::admission`'s registry at all.
use jammi_ai::fine_tune::adamw::AdamW;
use jammi_ai::fine_tune::optimizer::{clip_gradients, sorted_trainable_vars, ClipOutcome};

use crate::report::{FinetuneStepTier, Measurement};

use sha2::{Digest, Sha256};

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

/// A `--row-lengths` value that does not name a valid right-padded batch for
/// this run's `(batch, seq)` -- either the count does not match `batch`, or
/// some row's length is outside `1..=seq` (`0` is a REFUSAL in the B3-padded
/// arm's own guard inventory; `> seq` cannot describe a real padded row of a
/// `[batch, seq]` mask at all). A step this contract measures must be the
/// step it claims to measure, so a bad explicit value is refused rather than
/// silently building a mask that does not mean what the caller intended.
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidRowLengths(pub String);

impl std::fmt::Display for InvalidRowLengths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "--row-lengths is invalid: {}", self.0)
    }
}

impl std::error::Error for InvalidRowLengths {}

/// Refuse a `--row-lengths` whose shape cannot describe a real right-padded
/// `[batch, seq]` batch: wrong element count, or any entry outside
/// `1..=seq`. Never called when `row_lengths` is `None` (the ordinary dense
/// case) -- this only guards a value the operator explicitly supplied.
fn validate_row_lengths(
    lengths: &[usize],
    batch: usize,
    seq: usize,
) -> Result<(), InvalidRowLengths> {
    if lengths.len() != batch {
        return Err(InvalidRowLengths(format!(
            "--row-lengths named {} length(s) but --batch is {batch} -- one length per row is \
             required",
            lengths.len()
        )));
    }
    if let Some((row, &bad)) = lengths
        .iter()
        .enumerate()
        .find(|&(_, &l)| l == 0 || l > seq)
    {
        return Err(InvalidRowLengths(format!(
            "row {row}'s length {bad} is outside 1..={seq} -- 0 is a refusal in the B3-padded \
             arm's own guard inventory (every row needs at least one real token), and a length \
             above --seq {seq} cannot describe a real row of a [batch, seq] mask"
        )));
    }
    Ok(())
}

/// How many times this run's step loop invoked the production
/// [`clip_gradients`] — the counted fact backing the "clip on" A/B row,
/// rather than a log line an operator has to trust. Process-wide, so both
/// `run()` (which emits it as `FinetuneStepTier::clip_invocations`) and the
/// tests read it as a before/after delta the same way the fused-kernel
/// dispatch counters are read.
static CLIP_INVOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Snapshot the clip-invocation counter. `run()` takes one BEFORE its
/// untimed pre-step and one after the loop, and emits the delta on the
/// report (PR #381 audit B2: the counted fact was previously
/// `#[cfg(test)]`-readable only, so the emitted row carried the
/// `max_grad_norm` REQUEST but nothing counted); the invocation-count tests
/// below read the same delta around their own `run` call.
fn clip_invocations_snapshot() -> u64 {
    CLIP_INVOCATIONS.load(Ordering::Relaxed)
}

/// Attention-base op keys whose presence in `JAMMI_KERNELS_DISABLE`
/// (`kernels_disabled_requested`) means the OPERATOR asked for the eager
/// attention arm: the fused whole-attention-block CustomOp, the FA2 cascade
/// key that absorbs it on the flash branch, or the `"all"` wildcard
/// (`jammi_kernels::admission`'s module doc).
const ATTENTION_DISABLE_KEYS: [&str; 3] = ["attention_block", "attention_block_flash", "all"];

/// The attention REFERENCE CLASS the operator ASKED this run to measure —
/// the value `FinetuneStepTier::attention_arm` carries into the shared
/// jammi/torch identity check (see that field's doc and
/// `ci/scripts/perf/identity_fields.py`'s entry): `"eager"` iff an
/// attention base (`ATTENTION_DISABLE_KEYS`) is in
/// `kernels_disabled_requested`, else `"fused"`.
///
/// Deliberately NOT derived from the `attention_block_*_dispatches`
/// counters (PR #381 re-audit, class-A): those read EAGER whenever the
/// fused predicate declines BY DOMAIN — `head_dim != 64`, `seq > 4096`, a
/// dtype/contiguity/mask arm (`jammi_encoders::modernbert`'s predicate;
/// `report.rs`'s `attention_block_eager_dispatches` doc calls that reading
/// by-design) — so a legitimate jammi-fused leg on a non-64 `head_dim`
/// checkpoint would have read `"eager"`, mismatched torch's `"fused"`, and
/// INVALIDated the row over a MEASUREMENT, not a premise. Whether the fused
/// arm actually ran stays where it already lives: `fused_proof` and the
/// counters themselves. An identity field describes what was asked for;
/// a domain decline is a datum about the checkpoint, never an identity
/// mismatch. (The flash cascade is folded in for free: on the FA2 branch
/// `attention_block_flash` is the key the eager leg disables, and the
/// committed `s128_flash_on_1.json` fixture — block `0/0`, flash `840` —
/// reads `"fused"` here, where a counter derivation read `"none"`.)
fn attention_arm(kernels_disabled_requested: &[String]) -> &'static str {
    if kernels_disabled_requested
        .iter()
        .any(|k| ATTENTION_DISABLE_KEYS.contains(&k.as_str()))
    {
        "eager"
    } else {
        "fused"
    }
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
    /// runs entirely on device (`4n + 4` device ops for `n` trainable
    /// `Var`s: the squared-sum fold, the coefficient, and a
    /// `broadcast_mul` rescale per `Var`; zero `to_scalar`/`to_vec` calls),
    /// a device-op cost no `None` row can see. Recording both an on and an
    /// off row on the same box makes that cost a measured delta instead of
    /// an assumption.
    pub max_grad_norm: Option<f32>,
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
    /// Per-row REAL (non-pad) lengths for a genuinely right-padded batch —
    /// `lengths.len() == batch`, each `1 <= lengths[b] <= seq` (the B3-padded
    /// flash arm's own guard inventory, contract v4 item 1: `total == 0`
    /// -- every row length 0 -- is a REFUSAL in the ragged arm, and pooling
    /// needs at least one real token per row regardless). `None` (the
    /// default) is this tier's ORIGINAL, UNCHANGED behaviour: an all-ones
    /// dense mask built by [`Tensor::ones`], `step_once` calling
    /// `encoder.forward` (never `forward_with_lengths`) — see
    /// [`crate::report::FinetuneStepTier::row_lengths`]'s own doc for why
    /// this is the field's dense-leg IDENTITY value too (`[seq; batch]`),
    /// not merely a param default. `Some(lengths)` builds a genuine
    /// right-padded mask (row `b`'s first `lengths[b]` positions `1`, the
    /// rest `0` -- RIGHT padding, the prefix shape `jammi_encoders`' B3-
    /// padded flash arm validates -- see `build_fixture`'s `prefix_mask`)
    /// and routes every forward through
    /// [`jammi_encoders::ModernBert::forward_with_lengths`]'s trusted-
    /// lengths path P (contract v4 §3.7), the one production entry point
    /// that can reach the padded transport this leg exists to measure.
    /// `lengths` is a TRUST boundary exactly as `forward_with_lengths`'
    /// own doc describes: this tier does not re-derive lengths from a
    /// device-side mask reduction, it builds `mask` FROM `lengths`
    /// host-side, so the trust and the construction are the same act —
    /// there is no way for the two to disagree here the way an external
    /// caller's independently-sourced `lengths` could.
    pub row_lengths: Option<Vec<usize>>,
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
/// flow under test for that fix) as the code path a test drives. Also
/// returns the `VarMap` itself (cheap: `VarMap` is an `Arc<Mutex<..>>`
/// handle, `all_vars()`/`data()` both borrow rather than consume it) so a
/// test can read a NAMED trainable tensor's value directly — e.g. pinning
/// the optimizer's actual learning rate via the ZerosB-init `lora_b`
/// tensor's magnitude after exactly one step (see
/// `finetune_step_one_step_moves_lora_b_by_approximately_lr` below), which
/// nothing reachable through `FinetuneStepTier`'s own public fields could
/// otherwise verify.
#[allow(clippy::type_complexity)]
fn build_fixture(
    params: &FinetuneStepParams,
) -> Result<
    (
        jammi_encoders::ModernBert,
        AdamW,
        usize,
        Vec<Tensor>,
        Tensor,
        VarMap,
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

    let trainable = sorted_trainable_vars(&varmap);
    if trainable.is_empty() {
        return Err("no trainable LoRA tensors — target_modules matched nothing".into());
    }
    let trainable_count = trainable.len();
    let opt = AdamW::new(
        trainable,
        candle_nn::ParamsAdamW {
            lr: 2e-4,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    // `params.row_lengths` is `None` on every pre-existing call site (the
    // ORIGINAL dense-only behaviour, bit-identical, never routed through
    // `prefix_mask`): the mask stays the all-ones `Tensor::ones` this fixture
    // always built, and `step_once` (called with `row_lengths: None`, see
    // `run()`'s call sites) keeps calling `encoder.forward` -- never
    // `forward_with_lengths` -- exactly as before this field existed.
    let mask = match &params.row_lengths {
        None => Tensor::ones((params.batch, params.seq), DType::U32, &device)?,
        Some(lengths) => prefix_mask(lengths, params.seq, &device)?,
    };
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

    Ok((encoder, opt, trainable_count, blocks, mask, varmap))
}

/// Build a genuine RIGHT-padded `[batch, seq]` prefix mask from per-row
/// `lengths`: row `b`'s first `lengths[b]` positions are `1`, the rest `0`
/// -- the exact prefix shape `jammi_encoders`' `resolve_lengths_and_prefix`
/// trusts a `forward_with_lengths` caller to have built (that function's own
/// doc: "a caller whose `lengths` do NOT actually match `mask`'s real
/// padding structure gets a WRONG flash-eligibility decision, not a caught
/// error" -- this is the one place in this tier that owns keeping the two in
/// sync, by constructing `mask` FROM `lengths` rather than the reverse).
/// `lengths` is assumed already validated by [`validate_row_lengths`] (every
/// entry in `1..=seq`, `lengths.len() == batch`) -- called only from
/// `build_fixture`, downstream of that check in `run()`.
fn prefix_mask(lengths: &[usize], seq: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut host = vec![0u32; lengths.len() * seq];
    for (b, &len) in lengths.iter().enumerate() {
        for s in 0..len.min(seq) {
            host[b * seq + s] = 1;
        }
    }
    Tensor::from_vec(host, (lengths.len(), seq), device)
}

/// One forward + cosine-margin triplet loss + backward + (optional) clip +
/// optimizer step — the exact body [`run`]'s timed loop executes, factored
/// out so the SAME code can also be run once, untimed, before the timed loop
/// starts (see the pre-step call in `run()`), mirroring
/// `torch_finetune_step.py`'s own `_step_once` factoring for the identical
/// reason.
///
/// `max_grad_norm`: `Some(max_norm)` runs the PRODUCTION [`clip_gradients`]
/// after backward and before the optimizer step, at the same point in the
/// sequence the trainer does (`ctx.optimizer.set_learning_rate` →
/// [`jammi_ai::fine_tune::optimizer::clip_and_step`], which is
/// `clip_gradients` then `optimizer.step`). `None` skips clipping entirely
/// and is bit-identical to this function's behaviour before `max_grad_norm`
/// existed. `trainable` is the SAME `Var` list `opt` was built over
/// (`build_fixture`'s name-sorted `sorted_trainable_vars(&varmap)`) —
/// `clip_gradients` needs it to look up each `Var`'s gradient in the
/// `GradStore` `loss.backward()` returns, and folds the global norm in
/// exactly this order (see `run()`'s own note on why it must be sorted).
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
#[allow(clippy::too_many_arguments)]
fn step_once(
    encoder: &mut jammi_encoders::ModernBert,
    opt: &mut AdamW,
    blocks: &[Tensor],
    mask: &Tensor,
    batch: usize,
    batched_forward: bool,
    trainable: &[Var],
    max_grad_norm: Option<f32>,
    // `Some(lengths)` routes THIS call through `ModernBert::forward_with_lengths`
    // (path P, contract v4 §3.7) building a genuine right-padded prefix mask from
    // `lengths` instead of the dense all-ones mask `build_fixture` otherwise builds —
    // see `FinetuneStepParams::row_lengths`'s own doc. `None` (every existing call
    // site before this change) is bit-identical to this function's behaviour before
    // `row_lengths` existed.
    row_lengths: Option<&[usize]>,
) -> Result<f32, Box<dyn std::error::Error>> {
    let (a, p, n) = if batched_forward {
        // One forward over the concatenated groups, split after pooling —
        // the trainer's `encode_groups` shape.
        let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0)?;
        let joined_mask = Tensor::cat(&[mask, mask, mask], 0)?;
        let all = match row_lengths {
            // Anchor/positive/negative share the SAME per-row lengths (they
            // share `mask`, above) — concatenated three times in the SAME
            // row order as `joined`/`joined_mask` (group 0's `batch` rows,
            // then group 1's, then group 2's), so `joined_lengths[r]` names
            // the real length of `joined`'s row `r` exactly.
            Some(lengths) => {
                let joined_lengths: Vec<usize> = lengths
                    .iter()
                    .copied()
                    .cycle()
                    .take(lengths.len() * 3)
                    .collect();
                encoder.forward_with_lengths(&joined, &joined_mask, Some(&joined_lengths))?
            }
            None => encoder.forward(&joined, &joined_mask)?,
        };
        (
            all.narrow(0, 0, batch)?,
            all.narrow(0, batch, batch)?,
            all.narrow(0, 2 * batch, batch)?,
        )
    } else {
        match row_lengths {
            Some(lengths) => (
                encoder.forward_with_lengths(&blocks[0], mask, Some(lengths))?,
                encoder.forward_with_lengths(&blocks[1], mask, Some(lengths))?,
                encoder.forward_with_lengths(&blocks[2], mask, Some(lengths))?,
            ),
            None => (
                encoder.forward(&blocks[0], mask)?,
                encoder.forward(&blocks[1], mask)?,
                encoder.forward(&blocks[2], mask)?,
            ),
        }
    };
    let loss = triplet_loss(&a, &p, &n, 0.3)?;
    let mut grads = loss.backward()?;
    // Same point in the sequence the trainer clips at: after backward,
    // before the optimizer step (trainer.rs's `process_batch_loss` runs
    // `scaled_loss.backward()` then, at the accumulation boundary,
    // `clip_and_step` — clip_gradients then `optimizer.step` — never the
    // reverse). `None` skips this block entirely: bit-identical to the step
    // this function measured before `max_grad_norm` existed.
    if let Some(max_norm) = max_grad_norm {
        // `ClipOutcome` is `#[must_use]`: this tier asked for a clip over a
        // non-empty `trainable` with a validated `max_norm > 0`, so the ONLY
        // outcome that makes this a clip-on row is `Clipped`. `Disabled`
        // cannot happen (`validate_max_grad_norm` refused `<= 0` up front)
        // and `NoGradients` means the fixture's loss did not route through
        // any trainable `Var` — a broken measurement, not a datum; refuse
        // it rather than count an invocation that clipped nothing.
        let outcome = clip_gradients(trainable, &mut grads, max_norm as f64)
            .map_err(|e| format!("finetune-step clip_gradients: {e}"))?;
        match outcome {
            ClipOutcome::Clipped(_total_norm) => {
                // The norm tensor is deliberately NOT read back here: this
                // is the timed path, and the production trainer reads it
                // only on `refuse_nonfinite_norm`'s cadence, never per step.
            }
            other @ (ClipOutcome::Disabled | ClipOutcome::NoGradients) => {
                return Err(format!(
                    "finetune-step clip_gradients returned {other:?} for max_grad_norm={max_norm} \
                     over {} trainable Var(s) — the clip-on row would be claiming a clip that \
                     never ran (INVALID run, not a datum)",
                    trainable.len()
                )
                .into());
            }
        }
        CLIP_INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
    opt.step(&grads)?;
    // Force completion before returning: candle's CUDA queue is
    // asynchronous, so without this the caller's clock (when called from the
    // timed loop) would measure submission time, not execution time.
    Ok(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?)
}

/// Run the tier and return its report block.
///
/// Returns `Err` — not a `FinetuneStepTier` with a suspiciously-clean
/// dispatch split — when `JAMMI_KERNELS_DISABLE` named an op key that
/// never actually disabled a live dispatch this run (see the check just
/// before this function returns, and
/// `jammi_kernels::admission::unmatched_disables`'s doc): that is a typo
/// in the disable list, not evidence the forced-eager arm ran, and must
/// never be reported as a datum (contract K-aux). Also returns `Err`,
/// FIRST — before any device, checkpoint, or tensor work — when
/// `params.expect_kernels_disabled` is `Some` and does not match what this
/// process's `JAMMI_KERNELS_DISABLE` actually resolved to — see
/// `FinetuneStepParams::expect_kernels_disabled`'s doc.
pub fn run(params: &FinetuneStepParams) -> Result<FinetuneStepTier, Box<dyn std::error::Error>> {
    // Validate FIRST, before any device, checkpoint, or tensor work — a bad
    // explicit `--max-grad-norm` is a caller error, not something worth
    // paying for a build + warmup + measured steps to discover.
    if let Some(max_norm) = params.max_grad_norm {
        validate_max_grad_norm(max_norm)?;
    }
    // Same posture, for `--row-lengths`: a shape that cannot describe a real
    // right-padded `[batch, seq]` batch is a caller error, refused before
    // any device/checkpoint/tensor work -- never silently building a mask
    // that does not mean what the caller intended.
    if let Some(lengths) = &params.row_lengths {
        validate_row_lengths(lengths, params.batch, params.seq)?;
    }

    // `--expect-kernels-disabled` (contract K-aux, round 2 advisory): the
    // binary controls its own argv, so a caller that names its intended
    // `JAMMI_KERNELS_DISABLE` set on the SAME command line gets a hard
    // error instead of having to notice, after the fact, that the
    // process-observed `kernels_disabled_requested` in the emitted JSON
    // report came back empty (or different) — machine-enforced rather than
    // eyeballed. Compared as SORTED sets: `disabled_ops_requested()` is
    // already sorted, and the expectation is sorted here so the caller's
    // ordering on the command line is not load-bearing.
    //
    // Checked HERE, at the very top, before the checkpoint is even loaded
    // or a single tensor is built — unlike `unmatched_kernel_disables`
    // below (which genuinely cannot be checked before every dispatch site
    // has had its chance to fire this run), `disabled_ops_requested()` is
    // a pure function of the real `JAMMI_KERNELS_DISABLE` env var, resolved
    // once at first read and never dependent on anything this function
    // does afterward — so a mismatch here can fail fast, before paying for
    // a build + warmup + measured steps that were never going to produce a
    // valid report.
    if let Some(expected) = &params.expect_kernels_disabled {
        let mut expected_sorted = expected.clone();
        expected_sorted.sort();
        let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
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

    let device_label = match params.cuda_device {
        Some(o) => format!("cuda:{o}"),
        None => "cpu".to_string(),
    };

    // Base-checkpoint CONTENT identity — computed BEFORE `build_fixture`
    // loads the model, off the exact bytes this run reads, so it can never
    // drift from what the model actually built from (round-4 audit
    // fold-in on PR #372: the SAME mechanism `grad_oracle.rs`'s `run()`
    // uses, see this module's doc's determinant table).
    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&params.model_dir.join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&params.model_dir.join("model.safetensors"))?;

    let (mut encoder, mut opt, trainable_count, blocks, mask, varmap) = build_fixture(params)?;
    // The SAME `Var` list `opt` was built over (`build_fixture`'s own
    // `sorted_trainable_vars(&varmap)` — `Var` is a cheap `Arc`-backed
    // handle, so this is a fresh `Vec` over the identical underlying
    // storage, not a second fixture) — `step_once` needs it to look up each
    // `Var`'s gradient in the `GradStore` when `params.max_grad_norm` is
    // `Some`. NAME-SORTED, never `VarMap::all_vars()`'s raw `HashMap` order
    // (PR #381 audit B3, the esc-182 class): `clip_gradients` folds the
    // global norm left-to-right over THIS slice, so its order decides the
    // last bits of `total_norm` and of every clipped gradient — and
    // therefore of every clip-on `losses` entry this tier emits. A raw
    // `all_vars()` order is re-randomised per process by `HashMap`'s
    // hasher seed, which would make two same-seed invocations of this
    // binary disagree bit-for-bit; `clip_on_losses_are_bit_identical_
    // across_processes` below pins that they do not.
    let trainable = sorted_trainable_vars(&varmap);

    // The VRAM baseline is taken here, BEFORE the untimed pre-step below and
    // BEFORE the sampler starts — see the comment on `vram_baseline` for why
    // this snapshot is deliberately taken at a DIFFERENT point in the
    // sequence than the dispatch-counter "before" snapshots a few lines
    // down, which are taken AFTER the pre-step.
    //
    // `peak_vram_bytes` is measured via `nvidia-smi --query-gpu=memory.used`
    // (`device_memory_used_bytes` above), which is a DRIVER-level allocator
    // POOL high-water mark, not live-allocated bytes — it does NOT shrink
    // back down between steps (the same convention
    // `crates/jammi-kernels/artifacts/cuda-runs/2026-08-24-p1-softmax-fold-
    // bf8e807-a100-sxm4.json` (unification contract C8: this RECORD moved
    // here from its original pre-schema location under `crates/jammi-bench/`)
    // reasons about in 32 MiB pool blocks). For a monotone pool high-water, "baseline" means "before any
    // of this run's allocation happened" — i.e. right after the model,
    // optimizer, and fixture tensors are built (`build_fixture` above) but
    // before the untimed pre-step drives the pool up. If this baseline were
    // instead taken AFTER the pre-step, the pre-step's own allocation would
    // already have pushed `memory.used` up to (or near) the run's
    // high-water, and `VramSampler::finish`'s `peak.saturating_sub(baseline)`
    // would floor the reported delta at (or near) zero even though the run
    // legitimately uses many GB. Torch's counterpart
    // (`torch_finetune_step.py`) does not have this hazard because it reads
    // `max_memory_allocated() - memory_allocated()`, an allocator
    // live-bytes high-water that DOES recover between calls — so the two
    // stacks' baselines are deliberately taken at different points in their
    // respective step sequences in order to stay comparable under
    // `vram_delta(comparable)`.
    let vram_baseline = device_memory_used_bytes().unwrap_or(0);
    let sampler = VramSampler::start();

    // ONE untimed step, BEFORE the timed loop — mirrors
    // `torch_finetune_step.py`'s own untimed `_step_once` pre-step (see that
    // file's `run()` for the identical reasoning), so BOTH stacks discard
    // the SAME "update 0" before the officially reported `--warmup` step 0
    // begins. Without this, jammi's `losses[k]` was the loss after
    // `warmup+k` total optimizer updates while torch's `losses[k]` was the
    // loss after `warmup+k+1` updates (one update ahead) — `loss_first` was
    // the worst case, since `LoraInitMode::ZerosB` makes the LoRA delta
    // identically zero at construction, so jammi's un-fixed `loss_first` was
    // the PRISTINE (zero-optimizer-update) loss while torch's was already
    // one update in. With this pre-step, both stacks' `losses[k]` is the
    // loss after `warmup+k+1` total updates — see
    // `finetune_step_loss_first_is_the_post_pre_step_update` below, which
    // pins this exactly by driving this same `run()` entry point.
    //
    // Never timed, never appended to `losses`/`times`. The dispatch-counter
    // "before" snapshots immediately below are taken AFTER this call
    // returns, so this discarded step's own dispatch activity is excluded
    // from the measured delta. The VRAM sampler above, in contrast, IS
    // already running by this point (see above), so this step's allocation
    // legitimately contributes to the run's reported peak — it is part of
    // the same allocator pool the timed steps run in.
    //
    // The clip-invocation "before" snapshot is taken HERE, before the
    // pre-step — unlike the dispatch-counter snapshots below, which are
    // taken after it — so the emitted `clip_invocations` counts EVERY
    // `step_once` this run made (pre-step + warmup + measured), the same
    // window `torch_finetune_step.py`'s `clip_counter` covers. It is a
    // count of what ran, not a per-measured-step rate.
    let clip_invocations_before = clip_invocations_snapshot();
    step_once(
        &mut encoder,
        &mut opt,
        &blocks,
        &mask,
        params.batch,
        params.batched_forward,
        &trainable,
        params.max_grad_norm,
        params.row_lengths.as_deref(),
    )?;

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
    // Same mechanism, for the fused multi-tensor AdamW step kernel
    // (`jammi_ai::fine_tune::adamw::AdamW::step`, registry key
    // `"adamw_step_fused"` — the same key a caller names in
    // `JAMMI_KERNELS_DISABLE` to force the eager arm; see this tier's own
    // report doc for how that forced-eager run is validated end-to-end).
    let adamw_dispatch_before =
        jammi_kernels::admission::counters_for("adamw_step_fused").snapshot();
    // Same mechanism, for the FlashAttention-2 dense cascade (P6 Stage B
    // B3-dense) — a THREE-outcome snapshot (`fused`/`eager`/`declined`,
    // `jammi_kernels::admission::CascadeDispatchSnapshot`), not the
    // two-outcome shape the ops above use — see
    // `jammi_encoders::attention_block_flash_dispatch_snapshot`'s own doc.
    let attention_block_flash_dispatch_before =
        jammi_encoders::attention_block_flash_dispatch_snapshot();

    let mut times = Vec::with_capacity(params.steps);
    let mut losses = Vec::with_capacity(params.steps);
    for step in 0..(params.warmup + params.steps) {
        let t0 = Instant::now();
        // See `step_once`'s own doc for the loss-value placement convention
        // (PRE-update loss of this call's batch) and for the clip's
        // placement (after backward, before the optimizer step) — unchanged
        // by this refactor, just no longer duplicated inline here.
        let loss_val = step_once(
            &mut encoder,
            &mut opt,
            &blocks,
            &mask,
            params.batch,
            params.batched_forward,
            &trainable,
            params.max_grad_norm,
            params.row_lengths.as_deref(),
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
    let adamw_dispatch_after =
        jammi_kernels::admission::counters_for("adamw_step_fused").snapshot();
    let attention_block_flash_dispatch_after =
        jammi_encoders::attention_block_flash_dispatch_snapshot();

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

    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    // `losses` is pushed in lockstep with `times` above (same `if step >=
    // params.warmup` guard, same loop iteration), so it is never empty when
    // `times` is not — the same precondition `times[times.len() / 2]` above
    // already relies on (an `--steps 0` run panics there first).
    let loss_first = *losses.first().expect("losses populated alongside times");
    let loss_last = *losses.last().expect("losses populated alongside times");

    let tier = FinetuneStepTier {
        device: device_label,
        device_name: device_name(params.cuda_device),
        seed: params.seed,
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: params.lora_dropout as f64,
        // HARDCODED, unconditionally — see `FinetuneStepTier::margin`'s own
        // field doc: this tier has no `--margin` CLI flag, and the ONE call
        // site that uses this constant (`triplet_loss(&a, &p, &n, 0.3)`,
        // below) is not parameterized by it either — this field exists so
        // the VALUE is emitted for the leg-premise check, not so the run
        // itself becomes configurable this round.
        margin: 0.3,
        target_modules: params.target_modules.clone(),
        batched_forward: params.batched_forward,
        max_grad_norm: params.max_grad_norm,
        // IDENTITY (contract v4 §1's item 1, K7 audit: 17 -> 18 entries):
        // the per-row lengths this leg actually fed the encoder -- the
        // REQUESTED value when `--row-lengths` was supplied, or, on the
        // ORIGINAL dense default (`params.row_lengths == None`), the value
        // that describes the all-ones mask `build_fixture` built:
        // `[seq; batch]` (every row's real length equals `seq`, the SAME
        // discriminator `jammi_encoders`' `CompactedBatch::is_dense` uses:
        // `lengths.iter().all(|&l| l == seq)`). Never `null` -- both
        // producers always emit a concrete vector, so this field needs no
        // `FINETUNE_NULL_IS_A_VALUE_FIELDS` entry (unlike `max_grad_norm`).
        row_lengths: params
            .row_lengths
            .clone()
            .unwrap_or_else(|| vec![params.seq; params.batch]),
        trainable_tensors: trainable_count,
        warmup: params.warmup,
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
        adamw_fused_dispatches: adamw_dispatch_after
            .fused
            .saturating_sub(adamw_dispatch_before.fused),
        adamw_eager_dispatches: adamw_dispatch_after
            .eager
            .saturating_sub(adamw_dispatch_before.eager),
        // Counted over pre-step + warmup + measured (the "before" snapshot
        // sits above the pre-step, see there) — `warmup + steps + 1` on a
        // clip-on row, `0` on a clip-off one.
        clip_invocations: clip_invocations_snapshot().saturating_sub(clip_invocations_before),
        // What the operator ASKED for (the resolved `JAMMI_KERNELS_DISABLE`
        // request), never what the predicate measured — see `attention_arm`.
        attention_arm: attention_arm(&kernels_disabled_requested).to_string(),
        attention_block_flash_fused_dispatches: attention_block_flash_dispatch_after
            .fused
            .saturating_sub(attention_block_flash_dispatch_before.fused),
        attention_block_flash_declined_dispatches: attention_block_flash_dispatch_after
            .declined
            .saturating_sub(attention_block_flash_dispatch_before.declined),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        // The SAME function `report::Provenance::baked` calls for
        // `report.provenance.build_features` — never a second,
        // independently-drifting computation (unification contract C2.2;
        // see that function's own doc for why this tier ALSO carries its
        // own echo rather than relying solely on the `Report` wrapper).
        build_features: crate::report::build_features(),
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
    };
    // K7-completeness, enforced on every real run (unification contract
    // C3.1) — see `report::assert_identity_fields_present`'s own doc.
    let value = serde_json::to_value(&tier).expect("serialize FinetuneStepTier for self-check");
    crate::report::assert_identity_fields_present(&value, FinetuneStepTier::IDENTITY_FIELDS);
    Ok(tier)
}

/// The concrete device sub-class, so a recorded rate stays interpretable across
/// a heterogeneous rented fleet. `pub(crate)`: `grad_oracle.rs`'s own report
/// reuses this EXACT function (never a second `nvidia-smi --query-gpu=name`
/// call site that could drift from this one) — see that module's doc's
/// determinant table.
pub(crate) fn device_name(cuda_device: Option<usize>) -> String {
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

/// sha256 (hex-encoded) of `path`'s raw bytes, plus the byte length —
/// STREAMING (a bounded-size buffer, never `std::fs::read`'s whole-file
/// `Vec`): a real checkpoint's `model.safetensors` can be on the order of a
/// few GB (ModernBERT-large-class), and loading the entire file into
/// memory just to hash it would roughly double this tier's peak RSS for no
/// reason — the hasher only ever needs the CURRENT chunk. `pub(crate)`:
/// shared with `grad_oracle.rs`, which reuses this EXACT function (never a
/// second, independently-drifting hashing implementation) — see that
/// module's doc's determinant table and this tier's own
/// `checkpoint_config_sha256`/`checkpoint_weights_sha256` field docs.
pub(crate) fn sha256_and_len(
    path: &std::path::Path,
) -> Result<(String, u64), Box<dyn std::error::Error>> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| -> Box<dyn std::error::Error> { format!("opening {path:?}: {e}").into() })?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    // 64 KiB: large enough to amortize the syscall overhead of many small
    // reads, small enough that this function's own peak RSS contribution
    // stays negligible regardless of the file's total size.
    let mut buf = [0u8; 65536];
    let mut total_len: u64 = 0;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| -> Box<dyn std::error::Error> {
                format!("reading {path:?}: {e}").into()
            })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total_len += n as u64;
    }
    Ok((hex::encode(hasher.finalize()), total_len))
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

    /// Named `clip_tiny_params` (not `tiny_params`) to avoid colliding with
    /// this module's OTHER `tiny_params()` (the no-arg, `cookbook`-fixture
    /// helper the non-clip tests below use) — the two groups of tests were
    /// written independently against two different generic fixtures and
    /// this merge keeps both rather than forcing one to adopt the other's
    /// shape.
    fn clip_tiny_params(
        max_grad_norm: Option<f32>,
        steps: usize,
        warmup: usize,
    ) -> FinetuneStepParams {
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
            expect_kernels_disabled: None,
            row_lengths: None,
        }
    }

    #[test]
    fn clip_gradients_invocation_count_equals_measured_steps() {
        let _serial = CLIP_COUNTER_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = clip_invocations_snapshot();
        let params = clip_tiny_params(Some(1.0), 3, 0);
        let tier = run(&params).expect("finetune-step run with --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(tier.steps_measured, 3);
        // 1 (run()'s own untimed pre-step, added by the loss-first alignment
        // fix) + 3 (warmup + steps loop iterations, warmup=0) = 4: `run()`
        // clips at EVERY `step_once` call it makes, the discarded pre-step
        // included (see `step_once`'s own doc) — not just the ones that
        // land in `steps_measured`.
        assert_eq!(
            after - before,
            4,
            "clip_gradients must be invoked exactly once per step_once call \
             (pre-step plus every warmup/measured loop iteration), not logged \
             and trusted"
        );
    }

    #[test]
    fn clip_gradients_invocation_count_includes_warmup_iterations() {
        let _serial = CLIP_COUNTER_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = clip_invocations_snapshot();
        // warmup=1, steps=2: the trainer clips every step it runs, warmup
        // included (the loop that clips is the same loop that warms up), so
        // the counted fact must reflect all 3 loop iterations, not just the
        // 2 that land in `steps_measured`. Plus 1 more for `run()`'s own
        // untimed pre-step (see the sibling test above) => 4 total.
        let params = clip_tiny_params(Some(1.0), 2, 1);
        let tier = run(&params).expect("finetune-step run with --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(tier.steps_measured, 2);
        assert_eq!(
            after - before,
            4,
            "clip must run during the pre-step and warmup too"
        );
    }

    #[test]
    fn clip_gradients_never_invoked_when_max_grad_norm_absent() {
        let _serial = CLIP_COUNTER_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let before = clip_invocations_snapshot();
        let params = clip_tiny_params(None, 2, 0);
        run(&params).expect("finetune-step run without --max-grad-norm");
        let after = clip_invocations_snapshot();
        assert_eq!(
            after, before,
            "clip_gradients must not run at all when --max-grad-norm is absent"
        );
    }

    #[test]
    fn max_grad_norm_rejects_zero() {
        let params = clip_tiny_params(Some(0.0), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=0.0 must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_negative() {
        let params = clip_tiny_params(Some(-1.0), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=-1.0 must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_nan() {
        let params = clip_tiny_params(Some(f32::NAN), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=NaN must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    #[test]
    fn max_grad_norm_rejects_infinite() {
        let params = clip_tiny_params(Some(f32::INFINITY), 1, 0);
        let err = run(&params).expect_err("max_grad_norm=inf must be refused");
        assert!(
            err.downcast_ref::<InvalidMaxGradNorm>().is_some(),
            "must be the typed InvalidMaxGradNorm refusal, got: {err}"
        );
    }

    /// Runs the same construction `run` does (model + LoRA + AdamW, via
    /// [`build_fixture`]/[`step_once`] — the SAME primitives `run()` itself
    /// calls, never a second, independently-drifting reconstruction) for
    /// two forward/backward/(clip)/step iterations on CPU, and returns the
    /// final flattened trainable-parameter values. `run`'s own report cannot
    /// carry this signal — a step time is never a proxy for a parameter's
    /// bits — so the determinism tests below reconstruct the harness
    /// directly rather than reading it off [`FinetuneStepTier`].
    fn train_two_steps_and_flatten_params(max_grad_norm: Option<f32>) -> Vec<f32> {
        let params = clip_tiny_params(max_grad_norm, 2, 0);
        let (mut encoder, mut opt, _count, blocks, mask, varmap) =
            build_fixture(&params).expect("build fixture");
        let trainable = sorted_trainable_vars(&varmap);
        assert!(!trainable.is_empty(), "target_modules matched nothing");
        for _ in 0..2 {
            step_once(
                &mut encoder,
                &mut opt,
                &blocks,
                &mask,
                params.batch,
                params.batched_forward,
                &trainable,
                max_grad_norm,
                None,
            )
            .expect("step");
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

    /// A huge `max_norm` (1e9) matches the absent path bit-for-bit under the
    /// PRODUCTION device-side clip, which has no early return: `clip_coef =
    /// (max_norm / (total_norm + 1e-6)).min(1.0)` is computed and every
    /// gradient is multiplied by it unconditionally
    /// (`jammi_ai::fine_tune::optimizer::clip_gradients`'s exact
    /// bit-identity predicate). At `max_norm = 1e9` the measured
    /// `total_norm` from this tier's synthetic step is nowhere near that
    /// large, so `total_norm <= max_norm - 1e-6` holds by a huge margin,
    /// `clip_coef` clamps to EXACTLY `1.0`, and `x * 1.0 == x` for every
    /// finite `x` — the multiply is bit-identical to a no-op, not because it
    /// was skipped. The boundary this bit-identity predicate does NOT cover
    /// — `total_norm` within `1e-6` of `max_norm`, where `clip_coef` is
    /// strictly less than `1.0` and the rescale is NOT bit-identical to
    /// no-clip — is pinned in `jammi-ai`'s own test,
    /// `at_max_norm_boundary_coef_is_not_bit_identical_to_no_clip`
    /// (`crates/jammi-ai/src/fine_tune/optimizer.rs`), which controls the
    /// closed-form `total_norm`/`max_norm` inputs this tier's end-to-end
    /// synthetic step cannot.
    #[test]
    fn huge_max_grad_norm_matches_absent_on_host() {
        // `Some(1e9)` below invokes the PRODUCTION `clip_gradients` (through
        // `step_once`), which increments the process-wide `CLIP_INVOCATIONS`
        // counter the `clip_gradients_invocation_count_*`/`clip_gradients_
        // never_invoked_when_max_grad_norm_absent` tests above read as a
        // before/after delta. `cargo test` runs tests on multiple threads by
        // default, so this call must hold the SAME `CLIP_COUNTER_SERIAL`
        // mutex those tests do — without it, this test's own clip calls can
        // land inside another thread's before/after measurement window and
        // corrupt its delta (a real, observed flake: interleaved runs report
        // 6 invocations against an expected 4).
        let _serial = CLIP_COUNTER_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let absent = train_two_steps_and_flatten_params(None);
        let huge = train_two_steps_and_flatten_params(Some(1e9));
        assert_eq!(
            absent, huge,
            "clip_gradients's unconditional multiply by clip_coef must be \
             bit-identical to no-op when max_norm is far above the actual \
             grad norm (clip_coef clamps to exactly 1.0)"
        );
    }

    /// One un-clipped backward over the tiny fixture (the SAME forward
    /// `step_once` runs, `batched_forward` on), returning the name-sorted
    /// trainable `Var`s, the raw `GradStore`, and a HOST-side `f64`
    /// reference of the global L2 norm over exactly those gradients — the
    /// independent oracle the active-arm test and the cross-process
    /// determinism test below both need (the latter to prove its chosen
    /// `max_norm` actually puts the coefficient strictly below `1.0`).
    fn first_step_grads_and_host_norm(
        params: &FinetuneStepParams,
    ) -> (Vec<Var>, candle_core::backprop::GradStore, f64) {
        let (encoder, _opt, _count, blocks, mask, varmap) =
            build_fixture(params).expect("build fixture");
        let trainable = sorted_trainable_vars(&varmap);
        assert!(!trainable.is_empty(), "target_modules matched nothing");
        let joined = Tensor::cat(&[&blocks[0], &blocks[1], &blocks[2]], 0).expect("cat");
        let joined_mask = Tensor::cat(&[&mask, &mask, &mask], 0).expect("cat mask");
        let all = encoder.forward(&joined, &joined_mask).expect("forward");
        let b = params.batch;
        let (a, p, n) = (
            all.narrow(0, 0, b).expect("a"),
            all.narrow(0, b, b).expect("p"),
            all.narrow(0, 2 * b, b).expect("n"),
        );
        let loss = triplet_loss(&a, &p, &n, 0.3).expect("loss");
        let grads = loss.backward().expect("backward");
        let mut total_sq = 0.0f64;
        let mut present = 0usize;
        for var in &trainable {
            if let Some(g) = grads.get(var.as_tensor()) {
                present += 1;
                for x in g
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("to_vec1")
                {
                    total_sq += (x as f64) * (x as f64);
                }
            }
        }
        assert!(
            present > 0,
            "no trainable Var received a gradient — broken fixture"
        );
        (trainable, grads, total_sq.sqrt())
    }

    /// The ACTIVE clip arm, against the host reference (PR #381 advisory:
    /// this module's other clip tests run at `max_norm = 1e9` — where the
    /// coefficient clamps to exactly `1.0` and the rescale is a bit-exact
    /// no-op — or with the flag absent, so none of them ever exercised a
    /// coefficient strictly below `1.0` through the bench's own fixture).
    /// Picks `max_norm = total_norm / 2` off the host-side norm so `coef ≈
    /// 0.5`, runs the PRODUCTION `clip_gradients` over the name-sorted
    /// `Var`s exactly as `step_once` does, and checks every clipped element
    /// against `g · min(1, max_norm / (norm + 1e-6))` computed on the host
    /// in `f64`, within 4 `f32` ULPs — the same band `jammi-ai`'s own
    /// `multi_var_clip_matches_host_reference_on_cpu` uses. Non-vacuous
    /// control: at least one element must have CHANGED, or the arm did not
    /// bite.
    #[test]
    fn active_clip_matches_host_reference_at_coefficient_below_one() {
        let params = clip_tiny_params(None, 1, 0);
        let (trainable, mut grads, host_norm) = first_step_grads_and_host_norm(&params);
        assert!(
            host_norm.is_finite() && host_norm > 0.0,
            "host norm {host_norm}"
        );
        let max_norm = host_norm / 2.0;
        let host_coef = (max_norm / (host_norm + 1e-6)).min(1.0);
        assert!(
            host_coef < 1.0,
            "test setup: coefficient must be strictly below 1.0, got {host_coef}"
        );

        let before: Vec<(usize, Vec<f32>)> = trainable
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                grads.get(v.as_tensor()).map(|g| {
                    (
                        i,
                        g.flatten_all()
                            .expect("flatten")
                            .to_vec1::<f32>()
                            .expect("to_vec1"),
                    )
                })
            })
            .collect();

        let outcome = clip_gradients(&trainable, &mut grads, max_norm).expect("clip");
        let total_norm = match outcome {
            ClipOutcome::Clipped(t) => t.to_scalar::<f32>().expect("norm read"),
            other => panic!("expected Clipped, got {other:?}"),
        };
        let device_vs_host_norm = ((total_norm as f64) - host_norm).abs();
        assert!(
            device_vs_host_norm <= 4.0 * (host_norm as f32).abs() as f64 * f32::EPSILON as f64,
            "device total_norm {total_norm} vs host {host_norm}: off by {device_vs_host_norm}"
        );

        let mut changed = 0usize;
        for (i, orig) in &before {
            let after = grads
                .get(trainable[*i].as_tensor())
                .expect("gradient still present after clip")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("to_vec1");
            assert_eq!(orig.len(), after.len());
            for (j, (o, a)) in orig.iter().zip(&after).enumerate() {
                let expected = (*o as f64) * host_coef;
                let tol = 4.0
                    * (expected.abs() as f32).max(f32::MIN_POSITIVE) as f64
                    * f32::EPSILON as f64;
                assert!(
                    ((*a as f64) - expected).abs() <= tol,
                    "Var {i} elem {j}: clipped {a} vs host {expected} (orig {o}, coef {host_coef}), tol {tol}"
                );
                if a.to_bits() != o.to_bits() {
                    changed += 1;
                }
            }
        }
        assert!(
            changed > 0,
            "non-vacuous control: the active arm must have rescaled at least one element"
        );
    }

    /// `max_norm` for the cross-process determinism test below — small
    /// enough that the tiny fixture's first-step gradient norm is
    /// comfortably above it (the child ASSERTS that, off the host-side
    /// norm, so a fixture whose norm ever drops below this fails loudly
    /// instead of silently testing the clamp-to-1.0 no-op arm).
    const CROSS_PROCESS_MAX_NORM: f32 = 1e-3;
    const CROSS_PROCESS_CHILD_ENV: &str = "JAMMI_BENCH_CLIP_DETERMINISM_CHILD";

    /// PR #381 audit B3 — the cross-PROCESS determinism proof the PR body
    /// called "not yet run": two separate invocations of this test binary,
    /// same seed, `--max-grad-norm` active (coefficient strictly below
    /// `1.0`, asserted), must emit bit-identical `losses`. This is the
    /// esc-182 class made observable at the bench's own CLI-shaped entry
    /// point (`run()`): `clip_gradients` folds the global norm in the
    /// order its `trainable` slice arrives, and a raw `VarMap::all_vars()`
    /// order is re-randomised per process by `HashMap`'s hasher seed — so
    /// with `all_vars()` at `build_fixture`/`run` (as before this round)
    /// the last bits of every clipped gradient, and therefore of every
    /// loss after the first, differed between two launches of the same
    /// command. A single-process test cannot see this (one process, one
    /// hasher seed); this one re-executes itself as two child processes
    /// (`std::env::current_exe()`, filtered `--exact` to this test, with
    /// `CROSS_PROCESS_CHILD_ENV` set) and compares what they printed.
    /// Also checks the counted fact: `clip_invocations == steps + warmup
    /// + 1` in each child.
    #[test]
    fn clip_on_losses_are_bit_identical_across_processes() {
        const STEPS: usize = 3;
        const WARMUP: usize = 1;
        if std::env::var_os(CROSS_PROCESS_CHILD_ENV).is_some() {
            // CHILD: prove the clip is active, run the real entry point,
            // print the loss bits for the parent to compare.
            let params = clip_tiny_params(Some(CROSS_PROCESS_MAX_NORM), STEPS, WARMUP);
            let (_, _, host_norm) = first_step_grads_and_host_norm(&params);
            assert!(
                host_norm > (CROSS_PROCESS_MAX_NORM as f64) * 2.0,
                "child: first-step grad norm {host_norm} is not comfortably above \
                 CROSS_PROCESS_MAX_NORM {CROSS_PROCESS_MAX_NORM} — the clip would not bite"
            );
            let tier = run(&params).expect("child run");
            let bits: Vec<String> = tier
                .losses
                .iter()
                .map(|l| format!("{:08x}", l.to_bits()))
                .collect();
            println!("CLIP_DETERMINISM_LOSSES {}", bits.join(","));
            println!("CLIP_DETERMINISM_INVOCATIONS {}", tier.clip_invocations);
            println!("CLIP_DETERMINISM_ATTENTION_ARM {}", tier.attention_arm);
            return;
        }

        let exe = std::env::current_exe().expect("current_exe");
        // libtest's name for this fn is the module path WITHOUT the crate
        // prefix (`finetune_step::tests::…`), which `module_path!()` carries.
        let (_, module) = module_path!()
            .split_once("::")
            .expect("module_path has a crate prefix");
        let name = format!("{module}::clip_on_losses_are_bit_identical_across_processes");
        let run_child = |label: &str| -> (String, String, String) {
            let out = std::process::Command::new(&exe)
                .args(["--exact", &name, "--nocapture", "--test-threads=1"])
                .env(CROSS_PROCESS_CHILD_ENV, "1")
                .output()
                .expect("spawn child test process");
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            assert!(
                out.status.success(),
                "child {label} failed ({:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
                out.status
            );
            // libtest prints `test <name> ... ` WITHOUT a newline before the
            // test body's own first `println!` under `--nocapture`, so the
            // first marker sits mid-line: match by substring, not prefix.
            let grab = |key: &str| -> String {
                stdout
                    .lines()
                    .find_map(|l| l.split_once(key).map(|(_, v)| v.trim().to_string()))
                    .unwrap_or_else(|| panic!("child {label} printed no `{key}` line:\n{stdout}"))
            };
            (
                grab("CLIP_DETERMINISM_LOSSES "),
                grab("CLIP_DETERMINISM_INVOCATIONS "),
                grab("CLIP_DETERMINISM_ATTENTION_ARM "),
            )
        };
        let (losses_a, invocations_a, arm_a) = run_child("A");
        let (losses_b, invocations_b, arm_b) = run_child("B");
        assert_eq!(
            losses_a.split(',').count(),
            STEPS,
            "child A must report one loss per measured step: {losses_a}"
        );
        assert_eq!(
            losses_a, losses_b,
            "same seed, same --max-grad-norm, two processes: clip-on losses must be bit-identical \
             (A={losses_a} B={losses_b}) — a mismatch is the esc-182 HashMap-order class"
        );
        assert_eq!(
            invocations_a,
            (STEPS + WARMUP + 1).to_string(),
            "clip_invocations must count pre-step + warmup + measured"
        );
        assert_eq!(invocations_a, invocations_b);
        assert_eq!(
            arm_a, arm_b,
            "attention_arm must be a function of the run, not the process"
        );
        assert!(
            ["fused", "eager"].contains(&arm_a.as_str()),
            "a single-arm run must read fused or eager, got {arm_a}"
        );
    }

    /// The `attention_arm` derivation, pinned (see the tier field's doc):
    /// it is the OPERATOR'S REQUEST — an attention base in the resolved
    /// `JAMMI_KERNELS_DISABLE` list — never the counters' verdict.
    #[test]
    fn attention_arm_is_the_operators_request_not_the_predicates_verdict() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(attention_arm(&s(&[])), "fused");
        assert_eq!(attention_arm(&s(&["layer_norm_fused"])), "fused");
        assert_eq!(attention_arm(&s(&["attention_block"])), "eager");
        assert_eq!(attention_arm(&s(&["attention_block_flash"])), "eager");
        assert_eq!(attention_arm(&s(&["all"])), "eager");
        assert_eq!(
            attention_arm(&s(&["geglu_fused", "attention_block"])),
            "eager"
        );
    }

    /// The emitted `clip_invocations` is the SAME delta the invocation-count
    /// tests above read off the process-wide counter — `warmup + steps + 1`
    /// on a clip-on row, `0` on a clip-off one — and `attention_arm` is a
    /// single-arm value on this fixture.
    #[test]
    fn tier_emits_counted_clip_invocations_and_a_single_attention_arm() {
        let _serial = CLIP_COUNTER_SERIAL
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let on = run(&clip_tiny_params(Some(1.0), 2, 1)).expect("clip-on run");
        assert_eq!(on.clip_invocations, 2 + 1 + 1);
        assert_eq!(on.max_grad_norm, Some(1.0));
        let off = run(&clip_tiny_params(None, 2, 1)).expect("clip-off run");
        assert_eq!(off.clip_invocations, 0);
        assert_eq!(off.max_grad_norm, None);
        for tier in [&on, &off] {
            assert_eq!(tier.warmup, 1);
            assert_eq!(
                tier.attention_arm,
                attention_arm(&tier.kernels_disabled_requested),
                "attention_arm must be the resolved JAMMI_KERNELS_DISABLE request"
            );
            // THE CLASS-A REPRODUCTION (PR #381 re-audit): this fixture has
            // `head_dim = 16`, so the fused attention-block predicate
            // DECLINES BY DOMAIN and the counters read eager — yet nothing
            // was disabled, so the leg the operator asked for is the fused
            // one and must read "fused" (a counter-derived value read
            // "eager" here and would have INVALIDated a real non-64-head_dim
            // A/B row against torch-sdpa).
            assert!(tier.kernels_disabled_requested.is_empty());
            assert_eq!(tier.attention_arm, "fused");
            assert!(
                tier.attention_block_eager_dispatches > 0 && tier.attention_block_fused_dispatches == 0,
                "test premise: the tiny fixture must be a domain decline (eager counters), got fused={} eager={}",
                tier.attention_block_fused_dispatches,
                tier.attention_block_eager_dispatches
            );
        }
    }

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
            max_grad_norm: None,
            expect_kernels_disabled: None,
            row_lengths: None,
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
        // AdamW-side counter: every measured step (plus the untimed
        // pre-step, which is outside this test's before/after window) runs
        // one `AdamW::step` over every trainable `Var`, so this total is
        // non-zero on any run with at least one trainable tensor.
        let adamw_total =
            |t: &FinetuneStepTier| t.adamw_fused_dispatches + t.adamw_eager_dispatches;

        for (name, total_of) in [
            ("ln", ln_total as fn(&FinetuneStepTier) -> u64),
            ("rope", rope_total),
            ("softmax", softmax_total),
            ("geglu", geglu_total),
            ("attention_block", attention_block_total),
            ("adamw", adamw_total),
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
        let (mut o_encoder, mut o_opt, _count, o_blocks, o_mask, o_varmap) =
            build_fixture(&params).expect("oracle fixture");
        let o_trainable = sorted_trainable_vars(&o_varmap);
        // Call #1: the PRISTINE (zero prior updates) loss.
        let pristine_loss = step_once(
            &mut o_encoder,
            &mut o_opt,
            &o_blocks,
            &o_mask,
            params.batch,
            params.batched_forward,
            &o_trainable,
            None,
            None,
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
            &o_trainable,
            None,
            None,
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

    /// MUTATION-TRIAGE test (cargo-mutants caught `build_fixture`'s literal
    /// `lr: 2e-4` field being DELETED from the `ParamsAdamW` struct-update
    /// expression, silently falling back to `..Default::default()`'s `lr =
    /// 0.001` — 5x larger — completely undetected): no existing test reads
    /// the optimizer's actual learning rate; every test so far only
    /// compares LOSS VALUES produced by build_fixture, either against a
    /// SECOND independent build_fixture call (self-consistent even if BOTH
    /// silently used the wrong Default lr) or against a qualitative "not
    /// near margin" bound loose enough not to notice a 5x rate change.
    ///
    /// AdamW's own well-known first-step property is the oracle:
    /// [`jammi_ai::fine_tune::adamw::AdamW::step`]'s formula (`next_theta =
    /// theta*(1-lr*wd) - lr*(m_hat/(sqrt(v_hat)+eps))`, numerically
    /// identical to `candle_nn::AdamW::step`'s own — see that type's module
    /// doc) reduces, at `step_t ==
    /// 1`, to `m_hat == grad` and `v_hat == grad^2` exactly (bias
    /// correction cancels the `(1-beta1)`/`(1-beta2)` EMA blend on the
    /// very first update), so `m_hat/(sqrt(v_hat)+eps) == grad/(|grad| +
    /// eps) ≈ sign(grad)` for any `|grad| >> eps` (`eps` is `1e-8`,
    /// several orders below any real gradient this fixture produces). And
    /// since `LoraInitMode::ZerosB` makes `lora_b` start at EXACTLY `0.0`
    /// elementwise, the `theta*(1-lr*wd)` term is also exactly `0` on step
    /// 1 (`0 * anything == 0`) — so `next_theta ≈ -lr*sign(grad)`, meaning
    /// `max(|lora_b|)` after exactly one step is a direct, cheap read of
    /// "what lr did the optimizer actually apply", independent of the
    /// gradient's own scale. `2e-4` (the real value) is trivially
    /// distinguished from `0.001` (the mutant's 5x-larger Default
    /// fallback) or `0.0` (an lr that silently vanished) this way.
    #[test]
    fn finetune_step_one_step_moves_lora_b_by_approximately_lr() {
        let params = FinetuneStepParams {
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };
        let (mut encoder, mut opt, _count, blocks, mask, varmap) =
            build_fixture(&params).expect("build fixture");

        let lora_b_name = {
            let data = varmap.data().lock().expect("varmap mutex poisoned");
            data.keys()
                .find(|k| k.ends_with("lora_b"))
                .cloned()
                .expect("at least one lora_b tensor in the fixture's target_modules")
        };
        let read_lora_b = |varmap: &VarMap, name: &str| -> Vec<f32> {
            let data = varmap.data().lock().expect("varmap mutex poisoned");
            let var = data.get(name).expect("lora_b var");
            var.as_tensor()
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .expect("read lora_b")
        };

        let before = read_lora_b(&varmap, &lora_b_name);
        assert!(
            before.iter().all(|&v| v == 0.0),
            "lora_b must start at EXACTLY zero under LoraInitMode::ZerosB — {before:?}"
        );

        let trainable = sorted_trainable_vars(&varmap);
        step_once(
            &mut encoder,
            &mut opt,
            &blocks,
            &mask,
            params.batch,
            params.batched_forward,
            &trainable,
            None,
            None,
        )
        .expect("one step");

        let after = read_lora_b(&varmap, &lora_b_name);
        let max_abs_after = after.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        assert!(
            max_abs_after > 0.0,
            "lora_b did not move at all after one optimizer step -- lr silently zero?"
        );

        const LR: f32 = 2e-4;
        let relative_diff = (max_abs_after - LR).abs() / LR;
        assert!(
            relative_diff < 0.05,
            "max(|lora_b|) after one step = {max_abs_after}, expected ~{LR} (AdamW's step-1 \
             update is ≈lr per moved element under LoraInitMode::ZerosB) -- relative diff \
             {relative_diff} exceeds 5%; looks like the configured lr silently changed (e.g. \
             to ParamsAdamW::default()'s 0.001, 5x larger)"
        );
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught TWO of THREE comparison-
    /// operator mutations in `build_fixture`'s `(params.lora_dropout >
    /// 0.0).then_some(...)` boundary: `>` -> `==` and `>` -> `<`, both of
    /// which WRONGLY disable dropout for any POSITIVE `lora_dropout`
    /// value): builds two otherwise-identical fixtures differing ONLY in
    /// `lora_dropout` (0.5 vs 0.0, same seed), takes ONE untimed
    /// `step_once` on each (moving `lora_b` away from its `ZerosB`-init
    /// exact zero — see below for why this step is required), then
    /// compares a SECOND `step_once` call's loss between the two fixtures.
    ///
    /// The pre-step is NOT optional here: at the FRESH `ZerosB` init,
    /// `lora_b == 0` exactly, and the LoRA forward is `base(x) + scaling *
    /// dropout(x @ A^T @ B^T)` — with `B == 0`, `B^T` is the zero matrix,
    /// so the WHOLE LoRA term is `0` regardless of what dropout did to the
    /// `A^T` intermediate. This means the loss `step_once`'s FIRST call
    /// ever returns (the PRE-update loss of a pristine `B`) is
    /// mathematically INDEPENDENT of dropout — confirmed empirically: an
    /// earlier draft of this test compared FIRST-call losses directly and
    /// they were bit-identical (`0.26643285` both) even on the
    /// UN-mutated, correctly-dropout-engaging code, which is why this
    /// draft takes a pre-step first (mirrors `run()`'s own untimed
    /// pre-step reasoning, see `finetune_step_loss_first_is_the_post_pre_step_update`).
    /// After one update, `lora_b != 0`, so the SECOND call's forward DOES
    /// depend on dropout: `seed makes the A/B init and the dropout mask a
    /// pure function of the run` (`lora_linear.rs`'s own doc), so a
    /// correctly-engaged dropout at `p=0.5` changes the computation
    /// relative to the SAME seed's no-dropout computation; a mutant that
    /// wrongly leaves dropout disabled at 0.5 makes the two fixtures
    /// compute the IDENTICAL forward throughout, giving BIT-IDENTICAL
    /// losses even after the pre-step.
    ///
    /// LATTICE NOTE — the THIRD comparison-operator mutant at this site
    /// (`>` -> `>=`, which wrongly enables dropout AT EXACTLY `0.0`) is
    /// deliberately NOT tested and is not a gap: dropout at rate EXACTLY
    /// `0.0` drops nothing and scales by `1/(1-0.0) == 1` regardless of the
    /// mask, so "dropout engaged at p=0" and "dropout skipped entirely"
    /// are the mathematically IDENTICAL forward computation — no test
    /// could observe a difference, the same vacuous class this crate's
    /// `weight_decay: 0.01` (which happens to equal
    /// `ParamsAdamW::default().weight_decay`) already carries for its own
    /// deleted-field mutant (see `build_fixture`'s own comment).
    #[test]
    fn finetune_step_positive_lora_dropout_actually_changes_the_computation() {
        fn pre_step_then_loss(lora_dropout: f32) -> f32 {
            let mut params = tiny_params();
            params.lora_dropout = lora_dropout;
            let (mut encoder, mut opt, _count, blocks, mask, varmap) =
                build_fixture(&params).expect("fixture");
            let trainable = sorted_trainable_vars(&varmap);
            // Untimed pre-step: moves `lora_b` off its `ZerosB` exact
            // zero — see this test's own doc for why the FIRST call's
            // loss is provably dropout-independent regardless.
            step_once(
                &mut encoder,
                &mut opt,
                &blocks,
                &mask,
                params.batch,
                params.batched_forward,
                &trainable,
                None,
                None,
            )
            .expect("pre-step");
            step_once(
                &mut encoder,
                &mut opt,
                &blocks,
                &mask,
                params.batch,
                params.batched_forward,
                &trainable,
                None,
                None,
            )
            .expect("observed step")
        }

        let dropout_loss = pre_step_then_loss(0.5);
        let no_dropout_loss = pre_step_then_loss(0.0);

        assert_ne!(
            dropout_loss, no_dropout_loss,
            "lora_dropout=0.5 produced the IDENTICAL loss to lora_dropout=0.0 at the same \
             seed, post-pre-step -- looks like dropout was never actually engaged \
             (build_fixture's `> 0.0` boundary check regressed to always leaving dropout \
             disabled)"
        );
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught FIVE distinct arithmetic
    /// mutations inside `triplet_loss` itself — `a * p` -> `a + p`, `a *
    /// n` -> `a + n`, `neg - pos` -> `neg + pos`/`neg / pos`, `+ margin` ->
    /// `* margin` — ALL undetected): every other test that exercises
    /// `triplet_loss` does so only through `build_fixture`/`run`, either
    /// comparing losses against a SECOND independent call to the SAME
    /// (possibly identically-mutated) function, or checking a loose "not
    /// exactly margin" / "within a wide range" bound that a differently
    /// but still-bounded arithmetic expression can still clear. This test
    /// drives `triplet_loss` DIRECTLY (no candle model, no GPU) against a
    /// HAND-COMPUTED expected scalar for a tiny, exactly-representable-in-
    /// f32 fixture:
    ///
    /// `a = [[1,0],[0,1]]`, `p = [[1,0],[1,0]]`, `n = [[0,1],[0,1]]`,
    /// `margin = 0.3`:
    ///   `pos = sum(a*p, -1) = [1*1+0*0, 0*1+1*0] = [1.0, 0.0]`
    ///   `neg = sum(a*n, -1) = [1*0+0*1, 0*0+1*1] = [0.0, 1.0]`
    ///   `raw = relu(neg - pos + margin) = relu([0.0-1.0+0.3, 1.0-0.0+0.3])
    ///        = relu([-0.7, 1.3]) = [0.0, 1.3]`
    ///   `mean = (0.0 + 1.3) / 2 = 0.65`
    #[test]
    fn triplet_loss_matches_hand_computed_value() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let p = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (2, 2), &device).unwrap();
        let n = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 1.0], (2, 2), &device).unwrap();
        let loss = triplet_loss(&a, &p, &n, 0.3)
            .expect("triplet_loss")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(
            (loss - 0.65).abs() < 1e-5,
            "triplet_loss = {loss}, expected 0.65 (hand-computed, see this test's own doc)"
        );
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught THREE arithmetic
    /// mutations inside `synthetic_ids`'s own LCG step — `1 + (...)` ->
    /// `1 * (...)`, `>> 33` -> `<< 33`, and the `vocab - 1` subtraction ->
    /// `vocab / 1` — ALL undetected): the existing
    /// `finetune_step_synthetic_batch_offset_is_seed_plus_group_index_not_seed_times_group_index`
    /// test only pins the BLOCK-OFFSET arithmetic (`seed + i`) via a
    /// degeneracy signature, never the LCG's OWN internal formula, and
    /// `grad_oracle.rs`'s `batch_token_id_sums`-based test recomputes ids
    /// by calling THIS SAME function again (self-consistent, not
    /// independent). This test hand-computes the first 5 ids for `seed=1,
    /// vocab=10` (verified independently in Python using the identical
    /// 64-bit modular recurrence, masked to 64 bits at every step exactly
    /// like Rust's `wrapping_mul`/`wrapping_add`) and compares bit-for-bit.
    #[test]
    fn synthetic_ids_matches_hand_computed_lcg_sequence() {
        let device = Device::Cpu;
        let ids = synthetic_ids(1, 5, 10, 1, &device);
        let got: Vec<u32> = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(got, vec![6, 1, 6, 8, 7]);
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `device_name`'s whole
    /// function body being replaced with `String::new()`/`"xyzzy".into()`):
    /// no existing test reads `FinetuneStepTier::device_name`'s actual
    /// VALUE — only the pinned-key-set test in `report.rs` checks the key
    /// EXISTS, and that test constructs a literal fixture value rather
    /// than reading `device_name`'s real return.
    #[test]
    fn finetune_step_device_name_is_cpu_off_cuda() {
        let tier = run(&tiny_params()).expect("finetune-step run");
        assert_eq!(tier.device_name, "cpu");
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `run()`'s
    /// `s_per_step_mean` division `sum/len` surviving as `sum%len`/
    /// `sum*len`): wall-clock timings are real and non-reproducible, so
    /// this cannot hand-check an EXACT value — instead it pins a
    /// MATHEMATICAL INVARIANT that holds regardless of the actual
    /// timings: the mean of N positive reals is NEVER greater than their
    /// MAXIMUM, and `s_per_step_p50` (`times[len/2]`, which for
    /// `tiny_params()`'s `steps: 2` — exactly 2 samples — is the LARGER of
    /// the two, i.e. the max) IS that maximum. A `%`/`*` mutant breaks
    /// this invariant for realistic (sub-multi-second) per-step timings:
    /// `sum % len` returns ~`sum` unchanged when `sum < len` (true for any
    /// timing well under `len` seconds), which equals `2 * mean_correct >
    /// max(t1,t2) = p50`; `sum * len` inflates by `len^2` relative to the
    /// true mean, likewise `> p50` for any nonzero timing. The correct
    /// `sum/len` is provably `<= p50` (average never exceeds max).
    #[test]
    fn finetune_step_mean_never_exceeds_p50() {
        let tier = run(&tiny_params()).expect("finetune-step run");
        let mean = tier.s_per_step_mean.value.expect("mean measured");
        let p50 = tier.s_per_step_p50.value.expect("p50 measured");
        assert!(
            mean <= p50 + 1e-9,
            "s_per_step_mean ({mean}) exceeds s_per_step_p50 ({p50}) -- the mean of real, \
             sub-multi-second per-step timings can never exceed their maximum (p50, for a \
             2-sample steps:2 fixture); looks like the sum/len division was replaced with % \
             or *"
        );
    }

    /// MUTATION-TRIAGE test (cargo-mutants caught `run()`'s `steps_per_s =
    /// 1.0/p50` and `triplets_per_s = batch/p50` divisions surviving as
    /// `%`/`*`): pins the ALGEBRAIC IDENTITY each derived rate must
    /// satisfy (`rate * p50 == numerator`) rather than a raw wall-clock
    /// value, so this holds regardless of the actual (non-reproducible)
    /// timing.
    #[test]
    fn finetune_step_derived_rates_satisfy_their_defining_identity() {
        let params = tiny_params();
        let tier = run(&params).expect("finetune-step run");
        let p50 = tier.s_per_step_p50.value.expect("p50 measured");
        let steps_per_s = tier.steps_per_s.value.expect("steps_per_s measured");
        let triplets_per_s = tier.triplets_per_s.value.expect("triplets_per_s measured");

        let steps_identity = steps_per_s * p50;
        assert!(
            (steps_identity - 1.0).abs() < 1e-6,
            "steps_per_s ({steps_per_s}) * p50 ({p50}) = {steps_identity}, expected ~1.0 -- \
             looks like `1.0 / p50` was replaced with a non-division op"
        );

        let triplets_identity = triplets_per_s * p50;
        let expected = params.batch as f64;
        assert!(
            (triplets_identity - expected).abs() < 1e-6 * expected.max(1.0), // no-producer: degenerate-zero guard on an algebraic identity, not a tolerance.
            "triplets_per_s ({triplets_per_s}) * p50 ({p50}) = {triplets_identity}, expected \
             ~batch ({expected}) -- looks like `batch / p50` was replaced with a non-division \
             op"
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

    /// F1 REGRESSION (audit finding on PR #372): `peak_vram_bytes` is
    /// `VramSampler::finish`'s `peak.saturating_sub(baseline)`
    /// (`finetune_step.rs:218` at the time of writing). `saturating_sub`
    /// FLOORS at zero rather than wrapping — so if `baseline` is ever
    /// captured AT (or above) the run's own high-water mark, the reported
    /// delta collapses to zero even though the run legitimately allocated
    /// many GB. This pins the arithmetic directly, independent of
    /// `nvidia-smi`/a real GPU (`VramSampler`'s fields are plain atomics
    /// this test constructs directly, bypassing `start()`'s `nvidia-smi`
    /// precheck), using magnitudes drawn from a real A100 measurement of
    /// `main`'s (pre-regression) convention (b8-s512-d0.05,
    /// `peak_vram_bytes` = 14.98 GB) so the test is anchored to a
    /// production-scale number, not an arbitrary toy pair.
    ///
    /// This test cannot, by itself, prove `run()` captures `vram_baseline`
    /// BEFORE the untimed pre-step (that ordering fix has no CPU-observable
    /// effect: this box has no `nvidia-smi`, so `VramSampler::start()`
    /// returns `None` and `run()`'s `peak_vram_bytes` is
    /// `Measurement::not_yet_measured` regardless of ordering — see
    /// `finetune_step_peak_vram_bytes_is_not_yet_measured_off_gpu` below).
    /// The pod check that closes that gap is named in the PR's hand-off.
    #[test]
    fn vram_sampler_finish_reports_true_delta_not_floored_by_a_baseline_at_the_peak() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // `baseline`: model + optimizer resident, BEFORE any of this run's
        // allocation (the F1-fixed convention). `delta_bytes`: the exact
        // real-measurement magnitude for b8-s512-d0.05 (peak_vram_bytes =
        // 14.98 GB, tip 2c1a68d) so the asserted delta is traceable to a
        // live pod number, not an invented one.
        let baseline = 3 * GIB;
        let delta_bytes = 14_980_000_000_u64;
        let peak = baseline + delta_bytes;
        let sampler = VramSampler {
            peak: Arc::new(AtomicU64::new(peak)),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        };
        let m = sampler.finish(baseline);
        assert_eq!(
            m.value,
            Some((peak - baseline) as f64),
            "a baseline captured BEFORE this run's allocation must report the FULL delta, \
             not a floored/near-zero one"
        );
        assert!(
            m.value.unwrap() > 1.0e10,
            "expected a multi-GB delta (this pins the F1 class: a baseline mistakenly \
             captured AT the peak would floor this to ~0 via saturating_sub)"
        );

        // The bug this finding describes, reproduced in the arithmetic
        // alone: a baseline captured AT (or above) the peak — i.e. AFTER
        // the pool has already been driven to its high-water mark by an
        // untimed pre-step — floors to zero via `saturating_sub`, silently,
        // with no panic and no `None`.
        let collapsed_sampler = VramSampler {
            peak: Arc::new(AtomicU64::new(peak)),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        };
        let collapsed = collapsed_sampler.finish(peak); // baseline == peak
        assert_eq!(
            collapsed.value,
            Some(0.0),
            "sanity: this is the exact failure mode F1 describes — a same-or-later baseline \
             silently reports zero, never an error, which is precisely why the CALL-SITE \
             ordering in run() (fixed by this PR round) matters and cannot be caught by this \
             arithmetic test alone"
        );
    }

    /// Whether THIS box can drive `VramSampler` at all — the SAME probe
    /// `VramSampler::start()` gates on (`device_memory_used_bytes()`, an
    /// `nvidia-smi --query-gpu=memory.used` call), so the two lattice-cell
    /// tests below branch on exactly what `run()` branches on, never on a
    /// proxy (a CUDA feature flag, a hostname). `JAMMI_REQUIRE_CUDA` is the
    /// RED-on-demand hatch the crate's other device-gated tests use
    /// (`jammi-ai`'s `cuda_device`): with it set, the arm a box CANNOT
    /// observe is a hard failure, never a skip.
    fn vram_probe_present() -> bool {
        device_memory_used_bytes().is_some()
    }

    /// Lattice-cell arm for `run()`'s `peak_vram_bytes`, NO-`nvidia-smi`
    /// box (every CI lane): `VramSampler::start()` returns `None` (its
    /// first `device_memory_used_bytes()` call fails), so `run()`'s `match
    /// sampler { None => ... }` arm executes — pinned so a future change
    /// that panics or fabricates a value in the absent-GPU arm reddens
    /// immediately. On a box WITH `nvidia-smi` this arm is unobservable:
    /// the test says so and returns (the sibling
    /// `finetune_step_peak_vram_bytes_is_measured_on_a_box_with_nvidia_smi`
    /// asserts that box's arm), and — before PR #381's fix round — asserted
    /// `None` unconditionally, so it FAILED on every GPU box
    /// (`Some(0.0) != None`; seen on a100b at `main` 6d07b20 and at every
    /// PR tip since: the pod's `finetune_step::tests` leg could never be
    /// green, which is exactly what blocked #381's cuda-run artifact).
    #[test]
    fn finetune_step_peak_vram_bytes_is_not_yet_measured_off_gpu() {
        if vram_probe_present() {
            eprintln!(
                "finetune_step_peak_vram_bytes_is_not_yet_measured_off_gpu: nvidia-smi is present \
                 on this box, so the no-GPU arm is unobservable here — see the sibling \
                 `..._is_measured_on_a_box_with_nvidia_smi` test for this box's arm"
            );
            return;
        }
        let tier = run(&tiny_params()).expect("finetune-step run");
        assert_eq!(
            tier.peak_vram_bytes.value, None,
            "no nvidia-smi on this box => VramSampler::start() returns None => \
             peak_vram_bytes must be the not-yet-measured sentinel, never a fabricated 0.0"
        );
        assert_eq!(tier.peak_vram_bytes.unit, "bytes");
    }

    /// The OTHER lattice-cell arm: a box WITH `nvidia-smi` (a pod).
    /// `VramSampler::start()` returns `Some`, the sampler runs for the
    /// whole warmup+measured loop, and `run()`'s `Some(sampler) =>
    /// sampler.finish(vram_baseline)` arm emits a MEASURED, finite,
    /// non-negative device-total-minus-baseline delta — `Some(0.0)` is the
    /// honest reading for this CPU-resident tiny model (nothing of it lives
    /// on the device; the pool high-water never moves), never the
    /// not-yet-measured sentinel. Off a GPU box this arm is unobservable:
    /// with `JAMMI_REQUIRE_CUDA` set that is a hard failure (the
    /// RED-on-demand hatch, so a pod job that meant to prove this arm can
    /// never silently skip it); otherwise the test says so and returns.
    #[test]
    fn finetune_step_peak_vram_bytes_is_measured_on_a_box_with_nvidia_smi() {
        if !vram_probe_present() {
            assert!(
                std::env::var_os("JAMMI_REQUIRE_CUDA").is_none(),
                "JAMMI_REQUIRE_CUDA is set but nvidia-smi is not usable on this box — the \
                 on-GPU peak_vram_bytes arm cannot be proven here; a silent skip is not acceptable"
            );
            eprintln!(
                "finetune_step_peak_vram_bytes_is_measured_on_a_box_with_nvidia_smi: no nvidia-smi \
                 on this box, so the on-GPU arm is unobservable here — see the sibling \
                 `..._is_not_yet_measured_off_gpu` test for this box's arm"
            );
            return;
        }
        let tier = run(&tiny_params()).expect("finetune-step run");
        let v = tier
            .peak_vram_bytes
            .value
            .expect("nvidia-smi present => VramSampler ran => peak_vram_bytes must be measured");
        assert!(
            v.is_finite() && v >= 0.0,
            "measured peak_vram_bytes must be a finite non-negative delta, got {v}"
        );
        assert_eq!(tier.peak_vram_bytes.unit, "bytes");
    }

    // ─── row_lengths / padded-fixture knob (contract v4 §1 item 1) ──────────

    /// A4 (dense invariance): the DEFAULT (`row_lengths: None`, every
    /// existing call site before this field existed) reports the dense-leg
    /// IDENTITY value `[seq; batch]` -- see `FinetuneStepTier::row_lengths`'s
    /// own doc for why this is the field's dense value, not merely the param
    /// default. `tiny_params()` is `batch: 3, seq: 8`.
    #[test]
    fn row_lengths_defaults_to_the_dense_seq_vector_on_every_row() {
        let tier = run(&tiny_params()).expect("finetune-step run");
        assert_eq!(tier.row_lengths, vec![8, 8, 8]);
    }

    /// A4, the other half: every OTHER field on a dense (`row_lengths: None`)
    /// leg is untouched by this field's addition -- two runs of the
    /// identical dense params still agree bit-for-bit on `losses` (the same
    /// property `clip_on_losses_are_bit_identical_across_processes` already
    /// pins for the clip fields; this is the row_lengths-era re-proof that
    /// adding the field did not perturb the dense call path at all).
    #[test]
    fn row_lengths_absent_does_not_perturb_the_dense_leg() {
        let params = FinetuneStepParams {
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };
        let a = run(&params).expect("run 1");
        let b = run(&params).expect("run 2");
        assert_eq!(a.losses, b.losses);
        assert_eq!(a.row_lengths, vec![8, 8, 8]);
        assert_eq!(b.row_lengths, vec![8, 8, 8]);
    }

    /// `validate_row_lengths` refuses a count that does not match `--batch`.
    #[test]
    fn row_lengths_rejects_wrong_count() {
        let params = FinetuneStepParams {
            row_lengths: Some(vec![4, 4]), // tiny_params() batch is 3
            ..tiny_params()
        };
        let err = run(&params).expect_err("mismatched row_lengths count must be refused");
        assert!(
            err.downcast_ref::<InvalidRowLengths>().is_some(),
            "must be the typed InvalidRowLengths refusal, got: {err}"
        );
    }

    /// `validate_row_lengths` refuses a zero-length row (the B3-padded arm's
    /// own guard inventory: `total == 0` -- every row length 0 -- is a
    /// REFUSAL, never a silently-accepted empty row).
    #[test]
    fn row_lengths_rejects_a_zero_length_row() {
        let params = FinetuneStepParams {
            row_lengths: Some(vec![4, 0, 8]), // tiny_params() batch is 3, seq is 8
            ..tiny_params()
        };
        let err = run(&params).expect_err("a zero-length row must be refused");
        assert!(
            err.downcast_ref::<InvalidRowLengths>().is_some(),
            "must be the typed InvalidRowLengths refusal, got: {err}"
        );
    }

    /// `validate_row_lengths` refuses a length above `--seq` -- it cannot
    /// describe a real row of a `[batch, seq]` mask.
    #[test]
    fn row_lengths_rejects_a_length_above_seq() {
        let params = FinetuneStepParams {
            row_lengths: Some(vec![4, 4, 9]), // tiny_params() seq is 8
            ..tiny_params()
        };
        let err = run(&params).expect_err("a length above seq must be refused");
        assert!(
            err.downcast_ref::<InvalidRowLengths>().is_some(),
            "must be the typed InvalidRowLengths refusal, got: {err}"
        );
    }

    /// A genuinely padded, VALID `row_lengths` is accepted, routed through
    /// [`ModernBert::forward_with_lengths`]'s trusted-lengths path P
    /// end-to-end (a finite loss trajectory proves the forward/backward/step
    /// sequence completed, not just that the params were accepted), and
    /// reported back EXACTLY as requested -- the identity field is honest
    /// about what this leg actually fed the encoder, not a re-derived or
    /// rounded value.
    #[test]
    fn row_lengths_accepts_a_genuine_padded_batch_and_reports_it_back_exactly() {
        let params = FinetuneStepParams {
            row_lengths: Some(vec![4, 8, 2]), // tiny_params() batch is 3, seq is 8
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };
        let tier = run(&params).expect("finetune-step run over a genuinely padded batch");
        assert_eq!(tier.row_lengths, vec![4, 8, 2]);
        assert_eq!(tier.losses.len(), 1);
        assert!(
            tier.losses[0].is_finite(),
            "padded-batch loss must be finite, got {}",
            tier.losses[0]
        );
    }

    /// POSITIVE CONTROL: `row_lengths` actually changes what the encoder
    /// computes -- a genuinely padded batch (fewer real tokens averaged into
    /// the mean-pool per row) produces a DIFFERENT loss than the fully dense
    /// batch at the identical seed, proving the mask this tier built from
    /// `row_lengths` was actually consumed by the forward, not silently
    /// ignored (the same class of positive control
    /// `finetune_step_positive_lora_dropout_actually_changes_the_computation`
    /// exists for `lora_dropout`).
    #[test]
    fn row_lengths_padded_mask_actually_changes_the_computation_vs_dense() {
        let dense_params = FinetuneStepParams {
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };
        let dense = run(&dense_params).expect("dense run");

        let padded_params = FinetuneStepParams {
            row_lengths: Some(vec![2, 8, 4]), // NOT all == seq (8) -- genuinely padded
            warmup: 0,
            steps: 1,
            ..tiny_params()
        };
        let padded = run(&padded_params).expect("padded run");

        assert_ne!(
            dense.losses[0], padded.losses[0],
            "a genuinely padded row_lengths must change the pooled forward's loss relative to              the fully dense batch at the identical seed -- if this ever holds, the mask built              from row_lengths is not being consumed by the forward"
        );
    }

    /// `prefix_mask` builds the exact RIGHT-padded prefix shape
    /// `jammi_encoders::resolve_lengths_and_prefix`'s `trusted_lengths`
    /// branch trusts a `forward_with_lengths` caller to have built: row
    /// `b`'s first `lengths[b]` positions `1`, the rest `0` -- read back
    /// directly off the host, never inferred from a downstream forward's
    /// behaviour alone.
    #[test]
    fn prefix_mask_builds_the_exact_right_padded_prefix_shape() {
        let device = Device::Cpu;
        let lengths = [2usize, 0, 5];
        // `lengths` here intentionally includes a `0` -- `prefix_mask` itself
        // does not validate (that is `validate_row_lengths`'s job, already
        // proven above); this test only pins the SHAPE construction.
        let seq = 5usize;
        let mask = prefix_mask(&lengths, seq, &device).expect("build prefix mask");
        let host: Vec<u32> = mask
            .to_dtype(DType::U32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .expect("read mask");
        let expected: Vec<u32> = vec![
            1, 1, 0, 0, 0, // row 0: length 2
            0, 0, 0, 0, 0, // row 1: length 0
            1, 1, 1, 1, 1, // row 2: length 5 (== seq)
        ];
        assert_eq!(host, expected);
    }
}
