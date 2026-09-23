//! The jammi-vs-torch learning oracle: one forward+backward at identical
//! weights, compared by gradient direction, never by loss trajectory.
//!
//! ## Why gradients, not losses
//!
//! [`crate::finetune_step`] and the `train-step` ladder's torch edge over
//! it prove fused-vs-eager equivalence: same jammi build, one kernel path
//! forced on or off. That is evidence about fusion, not about learning —
//! if jammi's eager path itself computed a wrong gradient, both arms would
//! be wrong the same way. A jammi-vs-torch loss trajectory is not a
//! substitute either: even with matched optimizer-update placement and a
//! matched LoRA init distribution, the two frameworks draw different bits
//! for that distribution, and through a bf16 triplet hinge that alone
//! separates any multi-step trajectory permanently.
//!
//! ## The oracle this tier is
//!
//! Load the same base checkpoint on both stacks; load the same LoRA `A`/`B`
//! matrices from one shared file, so the init mismatch is removed at the
//! bit level; run one forward and backward on one identical synthetic
//! batch with LoRA dropout forced to zero; take no optimizer step; dump,
//! per trainable tensor by name, its gradient and the weight it was taken
//! at, both widened to `f32`.
//!
//! The dump is a `train-step` leg (`tiers.finetune_step`) whose
//! [`Measured::gradients`] is filled, filed under the `grads` take of the
//! `train-step` ladder's `torch -> reference` edge
//! ([`crate::ladder::definition::GRADIENTS_TAKE`]). The comparator judges
//! it as gradient agreement: per tensor, the weights must be the same bits
//! (a premise, not a tolerance), both gradients zero is vacuous, exactly
//! one zero or a non-finite entry breaks the structure, and a real pair's
//! cosine is held to a measured floor. The identity fields are
//! [`TrainStepPayload`]'s; the fields a single forward has no use for —
//! warmup, measured steps, dropout, clip — are free to differ from the
//! edge's timed repeats ([`crate::ladder::definition::GRADIENTS_TAKE_FREE_FIELDS`]).
//!
//! ## Weight interchange
//!
//! The shared file is a plain `safetensors` file written and read through
//! `candle_nn::VarMap::save`/`VarMap::load`, in jammi's own `VarBuilder`
//! path naming (`layer.3.Wqkv.lora_a`). A first invocation with no
//! `--lora-weights-in` uses its seeded init and writes that exact file
//! through `--lora-weights-out`; a second invocation loads it, and the two
//! agree bit for bit (`grad_oracle_self_consistency_round_trip`). The torch
//! peer, `crates/jammi-bench/reference/torch_grad_oracle.py`, translates
//! between PEFT's parameter names and jammi's in both directions; the
//! orientations already agree, so no transpose is involved.
//!
//! ## A single fresh-init call tests only `dL/dB`
//!
//! Under [`jammi_lora::LoraInitMode::ZerosB`] `B` starts at the exact zero
//! matrix, and `dL/dA` — routed through `B^T @ dL/d(output)` — is the exact
//! zero vector on both stacks whatever `A` is and whether or not either
//! backward is right. The comparator classifies such a pair as vacuous —
//! no evidence either way — rather than as agreement. Catching a real
//! `dL/dA` defect needs at least one optimizer step first; that extension
//! (teacher-forced: overwrite one side's weights with the other's after
//! each step) is not implemented.

use std::path::PathBuf;

use crate::finetune_step::{
    attention_arm, device_name, sha256_and_len, synthetic_ids, triplet_loss,
};
use crate::leg::{DispatchCounters, Facts, GradientTensor, Leg, Measured, Provenance};
use crate::report::{Measurement, TrainStepPayload};
use crate::rss::peak_rss_measurement;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;

/// Parameters the oracle drives its single forward+backward off of.
#[derive(Debug, Clone)]
pub struct GradOracleParams {
    pub model_dir: PathBuf,
    pub batch: usize,
    pub seq: usize,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub target_modules: Vec<String>,
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    pub cuda_device: Option<usize>,
    /// Drives the synthetic batch (`synthetic_ids(seed + i, ..)`, `i` in
    /// `0..3`) AND, when `lora_weights_in` is `None`, the fresh LoRA `A`
    /// draw (`LoraInitMode::ZerosB`, jammi's own SplitMix64 stream) — same
    /// role `FinetuneStepParams::seed` plays in `finetune_step.rs`.
    pub seed: u64,
    pub batched_forward: bool,
    /// A safetensors file (jammi's OWN internal `VarBuilder`-path naming —
    /// see this module's doc) to `VarMap::load` BEFORE the forward,
    /// overwriting the freshly-seeded LoRA `A`/`B` values in place. `None`
    /// keeps the fresh seeded draw (useful for the FIRST invocation that
    /// produces the shared file via `lora_weights_out`).
    pub lora_weights_in: Option<PathBuf>,
    /// Where to `VarMap::save` the LoRA weights ACTUALLY used for this
    /// forward (post-`lora_weights_in` load, if any) — lets a first
    /// invocation seed the shared file a second, independent invocation
    /// then loads via `lora_weights_in`.
    pub lora_weights_out: Option<PathBuf>,
}

/// The triplet margin the one forward is taken under: the margin
/// [`crate::finetune_step`] steps with, so a gradient leg and a timed leg
/// of the same edge state the same problem.
const MARGIN: f64 = 0.3;

/// The three synthetic groups — anchor, positive, negative — of one batch:
/// `synthetic_ids(.., seed + i, ..)` for `i` in `0..3`, the same blocks
/// both the batched and the per-group forward consume.
pub(crate) fn triplet_blocks(
    batch: usize,
    seq: usize,
    vocab_size: usize,
    seed: u64,
    device: &Device,
) -> Vec<Tensor> {
    (0..3)
        .map(|i| synthetic_ids(batch, seq, vocab_size, seed + i, device))
        .collect()
}

/// Run the oracle and return its leg. No optimizer step — see this
/// module's doc for why a gradient-direction comparison does not need one.
pub fn run(params: &GradOracleParams) -> Result<Leg<TrainStepPayload>, Box<dyn std::error::Error>> {
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

    // Base-checkpoint CONTENT identity — computed BEFORE the forward, off
    // the exact bytes this run loads, so it can never drift from what the
    // model actually built from (see this module's doc's determinant
    // table).
    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&params.model_dir.join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) = sha256_and_len(&weights)?;

    let mut varmap = VarMap::new();
    let empty_ranks = std::collections::HashMap::new();
    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &params.target_modules,
        layers_to_transform: &None,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        use_rslora: false,
        // Forced 0.0, unconditionally — a gradient-direction comparison
        // that let dropout differ between the two stacks would compare
        // gradients of DIFFERENT computations, not the same one through
        // two arithmetic paths. This tier has no CLI knob for this on
        // purpose.
        lora_dropout: None,
        rank_pattern: &empty_ranks,
        init_mode: jammi_lora::LoraInitMode::ZerosB,
        seed: params.seed,
        dropout_seed: params.seed,
    };

    let mut encoder = jammi_encoders::ModernBert::builder()
        .pooling(jammi_encoders::Pooling::Mean)
        .backbone_dtype(jammi_encoders::compute_precision_to_dtype(
            params.backbone_dtype,
        ))
        .lora(lora)
        .build(&[weights.as_path()], &config, &device, &varmap)?;
    // Training mode: dropout is forced off above (`lora_dropout: None`), so
    // this only selects the training-arm KERNEL COMPOSITION (fused
    // whole-attention-block / fused LoRA site where eligible), not any
    // additional randomness — the point is comparing the SAME arithmetic
    // shape jammi actually trains with, not the (dropout-free either way)
    // eval-mode composition.
    encoder.set_training(true);

    if let Some(path) = &params.lora_weights_in {
        varmap
            .load(path)
            .map_err(|e| -> Box<dyn std::error::Error> {
                format!(
                    "loading --lora-weights-in {path:?} into the VarMap failed: {e} — the file's \
                 tensor names must match jammi's OWN internal VarBuilder-path naming for this \
                 exact config (model dir / target-modules / lora-rank), see grad_oracle.rs's \
                 module doc's 'Weight interchange format' section"
                )
                .into()
            })?;
    }

    let trainable: Vec<Var> = {
        let data = varmap.data().lock().map_err(|_| "VarMap mutex poisoned")?;
        let mut named: Vec<(String, Var)> =
            data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named.into_iter().map(|(_, v)| v).collect()
    };
    if trainable.is_empty() {
        return Err("no trainable LoRA tensors — target_modules matched nothing".into());
    }
    // Names, same sorted order as `trainable`, kept alongside for the
    // gradient-dump loop below (a `Var` alone does not carry its own name).
    let trainable_names: Vec<String> = {
        let data = varmap.data().lock().map_err(|_| "VarMap mutex poisoned")?;
        let mut names: Vec<String> = data.keys().cloned().collect();
        names.sort();
        names
    };

    if let Some(path) = &params.lora_weights_out {
        varmap.save(path)?;
    }

    let mask = Tensor::ones((params.batch, params.seq), DType::U32, &device)?;
    let blocks = triplet_blocks(
        params.batch,
        params.seq,
        config.vocab_size,
        params.seed,
        &device,
    );

    // The dispatch counters around this one forward and backward alone.
    let dispatch_before = DispatchCounters::snapshot();

    let (a, p, n) = if params.batched_forward {
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
    let loss = triplet_loss(&a, &p, &n, MARGIN)?;
    let grads = loss.backward()?;
    let loss_val = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    let dispatch = DispatchCounters::snapshot().since(&dispatch_before);

    let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
    let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();

    let mut gradients = std::collections::BTreeMap::new();
    for (name, var) in trainable_names.into_iter().zip(trainable.into_iter()) {
        let tensor = var.as_tensor();
        let shape = tensor.dims().to_vec();
        let weight = tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let grad_tensor = grads.get(tensor).ok_or_else(|| {
            format!("no gradient recorded for trainable tensor {name:?} — did backward() reach it?")
        })?;
        let grad = grad_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        gradients.insert(
            name,
            GradientTensor {
                shape,
                grad,
                weight,
            },
        );
    }

    let arm = crate::kernel_arm::arm_label(&kernels_disabled_requested);
    let not_timed = || Measurement::not_yet_measured("s");
    let payload = TrainStepPayload {
        device: device_label,
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        seed: params.seed,
        batch: params.batch,
        seq: params.seq,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: 0.0,
        margin: MARGIN,
        target_modules: params.target_modules.clone(),
        batched_forward: params.batched_forward,
        max_grad_norm: None,
        trainable_tensors: gradients.len(),
        warmup: 0,
        row_lengths: vec![params.seq; params.batch],
        steps_measured: 0,
        losses: vec![loss_val],
        loss_first: loss_val,
        loss_last: loss_val,
        clip_invocations: 0,
        s_per_step_p50: not_timed(),
        s_per_step_mean: not_timed(),
        steps_per_s: Measurement::not_yet_measured("steps/s"),
        triplets_per_s: Measurement::not_yet_measured("triplets/s"),
    };
    let provenance = Provenance {
        device_name: device_name(params.cuda_device),
        build_features: crate::report::build_features()
            .into_iter()
            .map(str::to_owned)
            .collect(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        attention_arm: attention_arm(&kernels_disabled_requested).to_string(),
        arm: arm.to_string(),
        kernels_disabled_requested,
        kernels_disabled_fired,
        mutant: Default::default(),
        ran_on: None,
    };
    let measured = Measured {
        gradients: Some(gradients),
        peak_rss_bytes: peak_rss_measurement(),
        ..Default::default()
    };
    let facts = Facts {
        dispatch: Some(dispatch),
        ..Default::default()
    };
    let leg = Leg::new(payload, provenance, measured, facts);
    // Identity-field completeness, enforced on every real run.
    leg.to_value();
    Ok(leg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn tiny_model_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/tiny_modernbert_classifier")
    }

    fn gradients_of(
        leg: &Leg<TrainStepPayload>,
    ) -> &std::collections::BTreeMap<String, GradientTensor> {
        leg.measured.gradients.as_ref().expect("gradients measured")
    }

    fn tiny_params() -> GradOracleParams {
        GradOracleParams {
            model_dir: tiny_model_dir(),
            // 3, deliberately NOT 2: `run()`'s batched arm computes the
            // negative group's row offset as `2 * b`. At `b == 2`,
            // `2 * b == 2 + b == 4` — a MUTATION of that `*` to `+` is
            // undetectable by ANY test using `batch: 2` (the cargo-mutants
            // mutant `replace * with + in run` survives there). `b == 3` makes
            // `2 * b = 6` and `2 + b = 5` diverge.
            batch: 3,
            seq: 8,
            lora_rank: 2,
            lora_alpha: 4.0,
            target_modules: vec!["Wqkv".to_string()],
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            cuda_device: None,
            seed: 7,
            batched_forward: true,
            lora_weights_in: None,
            lora_weights_out: None,
        }
    }

    /// Drives the REAL entry point, `run()`. A fresh (no `lora_weights_in`)
    /// call must produce a finite, non-degenerate loss and at least one
    /// nonzero gradient entry — the cheap sanity floor every measurement
    /// tier in this crate applies before trusting a more specific claim.
    #[test]
    fn grad_oracle_run_produces_finite_loss_and_nonzero_gradients() {
        let leg = run(&tiny_params()).expect("grad-oracle run");
        assert!(leg.payload.loss_first.is_finite());
        assert_eq!(leg.payload.losses, vec![leg.payload.loss_first]);
        assert_eq!(leg.payload.steps_measured, 0);
        let gradients = leg.measured.gradients.as_ref().expect("gradients measured");
        assert!(!gradients.is_empty());
        assert_eq!(leg.payload.trainable_tensors, gradients.len());
        for (name, t) in gradients {
            assert_eq!(
                t.grad.len(),
                t.weight.len(),
                "{name}: grad/weight length mismatch"
            );
            assert_eq!(
                t.shape.iter().product::<usize>(),
                t.grad.len(),
                "{name}: shape does not match flattened grad length"
            );
            assert!(
                t.grad.iter().all(|g| g.is_finite()),
                "{name}: non-finite gradient entry"
            );
        }
        assert!(
            gradients.values().any(|t| t.grad.iter().any(|&g| g != 0.0)),
            "every gradient entry is exactly zero — looks like backward() never reached the \
             trainable tensors, or the dump read the wrong store"
        );
    }

    /// THE SELF-CONSISTENCY ROUND TRIP this module's doc promises: a
    /// SECOND, INDEPENDENT `run()` call that (a) loads the FIRST call's
    /// `lora_weights_out` dump via `lora_weights_in`, on the SAME
    /// synthetic batch (same `seed`), must reproduce the FIRST call's loss
    /// and every gradient entry BIT-FOR-BIT (CPU/F32 is deterministic —
    /// same weights in, same tokens in, same arithmetic). This is the
    /// mechanism proof the cross-framework comparison depends on: if the
    /// weight-interchange file did not carry the values that actually ran,
    /// this round trip would diverge even before torch enters the
    /// picture. Drives `run()` twice — the real entry point — never
    /// `VarMap::load`/`save` tested in isolation.
    #[test]
    fn grad_oracle_self_consistency_round_trip() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut first_params = tiny_params();
        first_params.lora_weights_out = Some(weights_path.clone());
        let first = run(&first_params).expect("first grad-oracle run");

        let mut second_params = tiny_params();
        second_params.lora_weights_in = Some(weights_path.clone());
        let second = run(&second_params).expect("second grad-oracle run");

        assert_eq!(
            first.payload.loss_first, second.payload.loss_first,
            "loss must round-trip bit-for-bit"
        );
        let (first, second) = (gradients_of(&first), gradients_of(&second));
        assert_eq!(
            first.keys().collect::<Vec<_>>(),
            second.keys().collect::<Vec<_>>(),
            "tensor name sets must match"
        );
        for (name, t1) in first {
            let t2 = &second[name];
            assert_eq!(
                t1.weight, t2.weight,
                "{name}: weight did not round-trip through the file"
            );
            assert_eq!(
                t1.grad, t2.grad,
                "{name}: gradient diverged despite identical weights+batch"
            );
        }

        let _ = std::fs::remove_file(&weights_path);
    }

    /// NEGATIVE CONTROL (family F: non-vacuous): loading a DIFFERENT LoRA
    /// weights file must actually change the dumped weight values (proving
    /// `--lora-weights-in` is not silently ignored) AND must actually
    /// change the gradient of the `lora_b` tensor too (proving the
    /// forward+backward actually ran against the loaded values, not the
    /// pre-load seeded draw). Two DIFFERENT seeds give two different fresh
    /// `A` draws (`B` starts at zero either way under
    /// `LoraInitMode::ZerosB`).
    ///
    /// This deliberately does NOT compare the `lora_a` tensor's gradient:
    /// under `LoraInitMode::ZerosB`, `dL/dA` is IDENTICALLY zero at a fresh
    /// (pre-optimizer-step) init regardless of `A`'s own value — the LoRA
    /// forward's `B @ (A @ x)` has `B == 0`, so the chain rule's `B^T @
    /// dL/d(output)` factor that backprops into `dL/dA` is the zero
    /// matrix. Asserting `lora_a`'s gradient differs would therefore be
    /// VACUOUS (0.0 != 0.0 never holds; the assertion would trivially pass
    /// for the wrong reason, or trivially fail always, neither of which
    /// tests the load path). `dL/dB`, in contrast, IS `A`-dependent even
    /// though `B == 0` — `dL/dB` is proportional to `(A @ x)`, which
    /// changes with `A` — so it is both meaningful (catches a silently
    /// skipped/wrong-path `varmap.load`) and cheap (no optimizer step
    /// needed to make it informative).
    #[test]
    fn grad_oracle_lora_weights_in_actually_overrides_the_fresh_init() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut seed_params = tiny_params();
        seed_params.seed = 123;
        seed_params.lora_weights_out = Some(weights_path.clone());
        let seeded = run(&seed_params).expect("seed run");

        let mut baseline_params = tiny_params();
        baseline_params.seed = 999; // a different fresh draw, no file loaded
        let baseline = run(&baseline_params).expect("baseline run (different fresh seed)");

        let mut loaded_params = tiny_params();
        loaded_params.seed = 999; // SAME batch/seed as baseline...
        loaded_params.lora_weights_in = Some(weights_path.clone()); // ...but weights overridden
        let loaded = run(&loaded_params).expect("loaded run");
        let (seeded, baseline, loaded) = (
            gradients_of(&seeded),
            gradients_of(&baseline),
            gradients_of(&loaded),
        );

        let any_name = baseline.keys().next().expect("at least one tensor").clone();
        assert_ne!(
            baseline[&any_name].weight, loaded[&any_name].weight,
            "lora_weights_in did not change the weight actually used -- looks like the load call \
             is being silently skipped or its error swallowed"
        );
        assert_eq!(
            loaded[&any_name].weight, seeded[&any_name].weight,
            "the loaded weight does not match the file's own recorded value"
        );

        // `any_name` is a `lora_a`-suffixed key (`.lora_a` < `.lora_b`
        // lexically, so `BTreeMap`'s first key is always `lora_a` here —
        // see this test's doc for why `lora_b`'s gradient, not `lora_a`'s,
        // is the informative one to assert on).
        assert!(
            any_name.ends_with("lora_a"),
            "fixture assumption broken: expected the sorted-first gradient key to be a \
             lora_a-suffixed name, got {any_name:?} -- update the lora_b lookup below to match"
        );
        // NOTE: `loaded` and `seeded` do NOT share a batch (`loaded_params`
        // keeps `seed = 999`, only its WEIGHTS come from the `seed = 123`
        // file), so their `lora_b` gradients are not expected to match —
        // only `baseline` (also `seed = 999`, no file loaded) is the right
        // same-batch comparator for the load-actually-took-effect check.
        let lora_b_name = format!("{}lora_b", any_name.strip_suffix("lora_a").unwrap());
        assert_ne!(
            baseline[&lora_b_name].grad, loaded[&lora_b_name].grad,
            "lora_weights_in changed the weight (asserted above) but NOT the lora_b gradient -- \
             looks like the forward+backward ran against the PRE-load seeded draw instead of the \
             loaded values (dL/dB is A-dependent even under LoraInitMode::ZerosB, see this test's \
             doc)"
        );

        let _ = std::fs::remove_file(&weights_path);
    }

    /// The three groups of a batch are `synthetic_ids(.., seed + i, ..)`
    /// for `i` in `0..3`, recomputed here independently of the helper: a
    /// self-consistent but wrong offset (`seed - i`, `seed * i`) would pass
    /// every other test in this module.
    #[test]
    fn triplet_blocks_are_synthetic_ids_at_seed_plus_i() {
        let params = tiny_params();
        let config_raw = std::fs::read_to_string(params.model_dir.join("config.json"))
            .expect("read config.json");
        let config: jammi_encoders::ModernBertConfig =
            serde_json::from_str(&config_raw).expect("parse config.json");
        let device = Device::Cpu;
        let blocks = triplet_blocks(
            params.batch,
            params.seq,
            config.vocab_size,
            params.seed,
            &device,
        );
        assert_eq!(blocks.len(), 3);
        for (i, block) in blocks.iter().enumerate() {
            let expected = synthetic_ids(
                params.batch,
                params.seq,
                config.vocab_size,
                params.seed + i as u64,
                &device,
            );
            assert_eq!(
                block.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
                expected.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
                "group {i} is not synthetic_ids(.., seed + {i}, ..)"
            );
        }
    }

    /// Mutation test (the cargo-mutants mutants turning the batched arm's
    /// `all.narrow(0, 2 * b, b)` — the negative group's row offset — into
    /// `2 + b`/`2 / b`): batched (one joined forward, split by
    /// `narrow`) and per-group (three separate forwards) MUST produce the
    /// identical loss/gradients for the SAME weights and the SAME
    /// synthetic batch — ModernBERT's per-row attention mask means no row
    /// can see any other row's tokens, so joining three groups into one
    /// forward is a pure reshape of the computation, not a different one.
    /// A miscomputed row offset in the batched arm picks the WRONG rows as
    /// one of anchor/positive/negative there, while the per-group arm
    /// (which uses no arithmetic at all — `blocks[0]`/`blocks[1]`/`blocks[2]`
    /// directly) stays correct, so the two arms diverge exactly when this
    /// arithmetic is wrong.
    #[test]
    fn grad_oracle_batched_and_unbatched_forward_agree() {
        let dir = tempdir();
        let weights_path = dir.join("lora_weights.safetensors");

        let mut seed_params = tiny_params();
        seed_params.lora_weights_out = Some(weights_path.clone());
        let _seeded = run(&seed_params).expect("seed run");

        let mut batched_params = tiny_params();
        batched_params.lora_weights_in = Some(weights_path.clone());
        batched_params.batched_forward = true;
        let batched = run(&batched_params).expect("batched run");

        let mut unbatched_params = tiny_params();
        unbatched_params.lora_weights_in = Some(weights_path.clone());
        unbatched_params.batched_forward = false;
        let unbatched = run(&unbatched_params).expect("unbatched run");

        // NOT bit-exact: candle's batched (3b-row) matmul kernel is free to
        // reduce in a different order than three separate b-row matmuls
        // (mathematically equivalent, not bitwise so — f32 addition is not
        // associative). MEASURED on this tiny fixture: relative differences
        // around 1e-6, four to five orders of magnitude below what a
        // genuine group-selection bug produces (a wrong `narrow` offset
        // picks ENTIRELY DIFFERENT rows, not a slightly-differently-rounded
        // version of the right ones) -- `TOL_REL`/`TOL_ABS` are generous
        // relative to the measured noise floor while staying far tighter
        // than a real defect would clear.
        const TOL_REL: f32 = 1e-3;
        const TOL_ABS: f32 = 1e-6;
        let (batched_loss, unbatched_loss) =
            (batched.payload.loss_first, unbatched.payload.loss_first);
        assert!(
            (batched_loss - unbatched_loss).abs()
                <= TOL_ABS + TOL_REL * batched_loss.abs().max(unbatched_loss.abs()),
            "batched vs per-group forward loss differs beyond floating-point reduction-order \
             noise: {} vs {} -- a group-selection offset bug (e.g. narrow(0, 2*b, b) \
             miscomputed) would silently pick the WRONG rows in the batched arm only",
            batched_loss,
            unbatched_loss
        );
        let (batched, unbatched) = (gradients_of(&batched), gradients_of(&unbatched));
        assert_eq!(
            batched.keys().collect::<Vec<_>>(),
            unbatched.keys().collect::<Vec<_>>()
        );
        for (name, t1) in batched {
            let t2 = &unbatched[name];
            assert_eq!(
                t1.weight, t2.weight,
                "{name}: weight differs between batched/unbatched runs (both loaded the SAME file)"
            );
            for (i, (&x, &y)) in t1.grad.iter().zip(t2.grad.iter()).enumerate() {
                let diff = (x - y).abs();
                let scale = x.abs().max(y.abs());
                assert!(
                    diff <= TOL_ABS + TOL_REL * scale,
                    "{name}[{i}]: batched={x} vs unbatched={y} (|diff|={diff}) exceeds the \
                     floating-point reduction-order noise tolerance (abs {TOL_ABS} + rel \
                     {TOL_REL}*{scale}) -- this looks like a real divergence (e.g. the WRONG \
                     rows selected), not rounding noise"
                );
            }
        }

        let _ = std::fs::remove_file(&weights_path);
    }

    /// A process-unique temp directory (never `/tmp` directly, and never
    /// shared with a parallel test in this same binary) for the two tests
    /// above that write real files.
    fn tempdir() -> PathBuf {
        // `cargo test`'s default thread pool runs this crate's `#[test]`
        // fns in PARALLEL — `std::process::id()` is shared by every test in
        // this binary, and `SystemTime::now()`'s actual clock resolution on
        // some platforms is coarser than true nanoseconds, so two
        // concurrently-running tests calling this within the same tick CAN
        // collide on path alone (MEASURED: this collision produced a
        // transient failure in `grad_oracle_lora_weights_in_actually_overrides_the_fresh_init`
        // while `grad_oracle_batched_and_unbatched_forward_agree` was
        // concurrently writing/reading the SAME nominally-"unique" path).
        // A process-wide atomic counter is monotonically unique per call
        // regardless of clock resolution, closing that race.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "jammi-bench-grad-oracle-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            n
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Every identity field of the `train-step` workload is present, and
    /// non-null where declared so, on a real leg `run()` produced — the
    /// same leg a timed step emits, so the two are comparable on the edge.
    #[test]
    fn grad_oracle_legs_carry_every_train_step_identity_field() {
        use crate::leg::Payload;
        let leg = run(&tiny_params()).expect("grad-oracle run");
        let value = leg.to_value();
        let obj = value.as_object().expect("object");
        for (field, nullable) in TrainStepPayload::IDENTITY_FIELDS {
            let entry = obj
                .get(*field)
                .unwrap_or_else(|| panic!("IDENTITY_FIELDS names {field:?}, absent on the leg"));
            if *nullable == crate::report::Nullable::NonNull {
                assert!(
                    !entry.is_null(),
                    "{field:?} is declared NonNull but serialized as null"
                );
            }
        }
        assert_eq!(obj["warmup"], 0);
        assert_eq!(obj["steps_measured"], 0);
        assert_eq!(obj["lora_dropout"], 0.0);
        assert!(obj["max_grad_norm"].is_null());
        assert!(obj["gradients"].is_object());
    }
}
