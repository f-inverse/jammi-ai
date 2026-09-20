//! CUDA bench legs for the padded (ragged flash) transport: the padded-shape
//! block-arm VRAM baseline and the padded loss-sequence flash-vs-block A/B.
//!
//! Both cases spawn the compiled `jammi-bench` binary as a fresh child
//! PROCESS — the same process-isolation discipline
//! `finetune_step_kernel_disable.rs`'s own module doc explains
//! (`jammi_kernels::admission`'s registries are process-wide `OnceLock`s).
//!
//! The whole file compiles only under the `live-gpu-tests` feature (CUDA
//! device 0); a spawned run that cannot open the device fails, it does not
//! pass vacuously. The two legs need different preconditions:
//!
//! 1. CUDA presence — the `live-gpu-tests` feature.
//! 2. Flash COMPILATION — `jammi_kernels::admission::FLASH_COMPILED`
//!    (`cfg!(feature = "flash-attn")` on `jammi-kernels`), reached from this
//!    crate through its `flash-attn` passthrough feature.
//! 3. Compute-capability ARCH — a MEMBER of
//!    `jammi_kernels::admission::flash_validated_arches()` (the SUBSET of
//!    `build.rs`'s compiled `-gencode` set with a green parity leg —
//!    sm80/86/89/90: a NARROWER concept than "compiled", never read
//!    directly from the compiled set), mirroring `jammi-encoders::modernbert`'s own `flash_arch_ok`
//!    (`flash_validated_arches().contains(&probe_cuda_compute_capability(device))`)
//!    — NOT `jammi_kernels::admission::ComputeCapability::meets_minimum`
//!    (a `>=` check, which would ALSO admit a hypothetical future arch
//!    this build never compiled a cubin for). The flash cascade's own
//!    domain predicate (`flash_arch_ok`'s call site,
//!    `"arch_in_flash_validated_set"`) declines on anything outside that
//!    enumerated set, regardless of compilation — a device this crate has
//!    no validated cubin for is NOT flash-eligible.
//!
//! [`a5_padded_block_arm_vram_baseline_leg`] needs ONLY precondition 1: it
//! disables `attention_block_flash` via the env-var registry, and
//! `admit_cascade`'s own `op_is_disabled` check fires FIRST, before any
//! `FLASH_COMPILED`/arch predicate is even consulted (`decide_flash_
//! admission`'s own doc) — so the block arm's decline-count assertion holds
//! on ANY CUDA host, flash-compiled or not, sm80 or not.
//!
//! [`a3_padded_loss_sequence_flash_vs_block_ab`] needs ALL THREE: its
//! `flash` leg asserts `attention_block_flash_fused_dispatches > 0`, which
//! requires the flash arm to actually be ELIGIBLE to fuse — compiled in AND
//! running on a device in the flash build's validated arch set — not merely
//! that a CUDA device exists. It compiles only with `flash-attn`; on a device
//! outside the validated arch set its Fused-dispatch assertion fails.
//!
//! Fixture: the SAME committed `head_dim == 64` checkpoint
//! `finetune_step_kernel_disable.rs` uses
//! (`crates/jammi-encoders/tests/fixtures/tiny_modernbert_head64`) —
//! `AttentionBlockFused`'s fixed domain (`ATTENTION_BLOCK_HEAD_DIM == 64`)
//! and the FA2 dense/padded cascade both require it. `--row-lengths 3,6`
//! (row 0 padded: 3 of 6 real tokens; row 1 dense: all 6 real) is a
//! genuinely padded batch by the encoder's own discriminator
//! (`lengths.iter().all(|&l| l == seq)` is FALSE here) and satisfies the
//! padded flash arm's guards (right-padded prefix, every row length >= 1).

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

/// Shared padded-fixture command: `head_dim == 64`, `bf16` (the FA2
/// cascade's fixed dtype domain), a genuinely padded `--row-lengths`, and
/// `--cuda 0`. `steps`/`warmup` deliberately small (`1`/`0`) — this is a
/// correctness/dispatch-shape leg, not a throughput sweep; a committed
/// artifact uses `stacked_sweep.sh`'s own step/warmup counts.
fn padded_command(model_dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args([
        "finetune-step",
        "--model-dir",
        &model_dir.to_string_lossy(),
        "--batch",
        "2",
        "--seq",
        "6",
        "--steps",
        "1",
        "--warmup",
        "0",
        "--lora-rank",
        "2",
        "--target-modules",
        "Wqkv,Wo",
        "--backbone-dtype",
        "bf16",
        "--cuda",
        "0",
        "--row-lengths",
        "3,6",
    ]);
    cmd
}

fn run_report(cmd: &mut Command) -> serde_json::Value {
    let output = cmd.output().expect("spawn jammi-bench finetune-step");
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("invalid JSON report: {e}\n{stdout}"))
}

/// **The padded-shape BLOCK-ARM VRAM baseline leg.** The flash
/// cascade is disabled (`JAMMI_KERNELS_DISABLE=attention_block_flash`), so
/// the padded batch runs the block/eager attention arm directly over the
/// FULL `[batch, seq, hidden]` mask — no compaction, no transport — the
/// comparator a padded-flash-vs-padded-block VRAM/step-time ratio
/// needs on the OTHER side: a padded-flash artifact has nothing to compare
/// against without this baseline.
///
/// Needs precondition 1 (CUDA presence) ONLY — see the module doc. This
/// leg never asserts a live flash Fused dispatch (it disables the flash
/// arm and asserts the DECLINE side), so it runs on any
/// CUDA host regardless of whether `jammi-kernels` was compiled with
/// `flash-attn` or the device's arch is a member of the flash cascade's
/// compiled set.
#[test]
fn a5_padded_block_arm_vram_baseline_leg() {
    let dir = model_dir();
    let report =
        run_report(padded_command(&dir).env("JAMMI_KERNELS_DISABLE", "attention_block_flash"));
    let tier = &report["tiers"]["finetune_step"];

    assert_eq!(
        tier["row_lengths"],
        serde_json::json!([3, 6]),
        "tier={tier}"
    );
    assert_eq!(
        tier["kernels_disabled_requested"],
        serde_json::json!(["attention_block_flash"]),
        "tier={tier}"
    );
    assert_eq!(
        tier["kernels_disabled_fired"],
        serde_json::json!(["attention_block_flash"]),
        "the disable must have actually fired a live dispatch this run, or the run is INVALID \
         (finetune_step::run's own unmatched_disables check already refuses that case — this \
         assertion is the leg's own belt-and-suspenders read of the SAME fact); tier={tier}"
    );
    assert_eq!(
        tier["attention_block_flash_fused_dispatches"].as_u64(),
        Some(0),
        "the flash arm was disabled — it must never dispatch Fused; tier={tier}"
    );
    assert!(
        tier["attention_block_flash_declined_dispatches"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "the flash arm was disabled — every admission attempt must have DECLINED (via \
         op_disabled, consulted first, before any predicate work); tier={tier}"
    );
    let peak_vram = tier["peak_vram_bytes"]["value"]
        .as_f64()
        .expect("peak_vram_bytes must be measured on a CUDA leg");
    assert!(
        peak_vram.is_finite() && peak_vram >= 0.0,
        "measured peak_vram_bytes must be a finite non-negative delta, got {peak_vram}"
    );
    let step_p50 = tier["s_per_step_p50"]["value"]
        .as_f64()
        .expect("s_per_step_p50 must be measured");
    assert!(
        step_p50.is_finite() && step_p50 > 0.0,
        "measured s_per_step_p50 must be finite and positive, got {step_p50}"
    );
}

/// **The padded loss-sequence flash-vs-block A/B.** Two legs over the
/// IDENTICAL padded fixture/seed: `flash` (the flash cascade admitted,
/// the padded arm) and `block` (the shipping explicit disable form —
/// `JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused`, NEVER the
/// `=all` wildcard, which would also disable ops this A/B does not compare).
///
/// Three notes on reading this leg's results:
///
/// (a) A GREEN result on this leg proves the padded TRANSPORT dispatches and
///     produces finite losses on both arms — nothing about whether the two
///     arms' loss VALUES agree.
/// (b) GRADIENT-GRAPH ASYMMETRY: the flash (padded/ragged) arm repads its
///     compacted output back to `[batch, seq, hidden]` with ZEROED pad
///     rows before continuing the layer stack — those repad-zeros SEVER
///     the pad rows' gradient path relative to the block arm (which never
///     compacts, so every row, pad or real, stays live in the autograd
///     graph throughout). No LIVE loss this fixture computes actually
///     consumes a pad row's gradient (`ner_loss` is latent — nothing
///     constructs `TrainingBatch::Ner` — and every pooling verb this tier
///     uses masks pad rows out before the loss), so this asymmetry is not
///     itself an observable defect on this leg, but it means a bit-level
///     gradient comparison between the two arms would NOT be expected to
///     agree on pad-row gradients even in a fully correct implementation —
///     never read a divergence there as a bug without checking this first.
/// (c) ASYMMETRIC PER-FORWARD SYNC COST, attributed here rather than left
///     as an unstated confound inside either arm's step-time number: this
///     leg's `flash` arm passes `Some(row_lengths)` into
///     `forward_with_lengths` (the trusted-lengths path), which does not
///     trust `is_prefix` on faith. Every time
///     the flash admission cascade actually reaches
///     `resolve_lengths_and_prefix` (i.e. every forward on this leg's
///     `flash` arm, which clears every cheaper capability/domain gate
///     first), `trusted_lengths_agree_with_mask` pays exactly ONE device
///     reduction + ONE D2H `to_vec1` sync — the SAME sync class the
///     mask-derived path's `compute_lengths_and_prefix` already pays, per that function's own
///     doc — to prove the host-supplied `row_lengths` really agrees with
///     the device-side mask before `is_prefix = true` is ever returned.
///     This leg's `block` arm pays NONE of that: `JAMMI_KERNELS_DISABLE=
///     attention_block_flash` makes `admit_cascade`'s `op_is_disabled`
///     check fire FIRST, before
///     `flash_admission_predicate`/`resolve_lengths_and_prefix` is ever
///     invoked, so the mask-agreement sync is never reached. The two arms'
///     step-time numbers are therefore NOT sync-symmetric: the `flash` arm
///     carries one extra device reduction + D2H round-trip per forward that
///     the `block` arm structurally cannot pay. This is not a benchmark
///     artifact to subtract out — the validation is part of the flash arm's
///     real production cost on the public `forward_with_lengths` edge (it
///     exists to catch a lying `trusted_lengths` before it drives
///     `unpad_gather_indices`/`repad_rows`) — so it rides inside the flash
///     arm's measured `s_per_step_p50` by design; a reader comparing the
///     two arms' step times must account for it explicitly rather than
///     assume both pay the same mask-path sync cost.
#[cfg(feature = "flash-attn")]
#[test]
fn a3_padded_loss_sequence_flash_vs_block_ab() {
    let dir = model_dir();

    let flash_report = run_report(&mut padded_command(&dir));
    let flash_tier = &flash_report["tiers"]["finetune_step"];
    assert_eq!(
        flash_tier["attention_block_flash_declined_dispatches"].as_u64(),
        Some(0),
        "a VALID flash-arm timing leg must read 0 declined (bench masks are prefix by \
         construction) — tier={flash_tier}"
    );
    assert!(
        flash_tier["attention_block_flash_fused_dispatches"]
            .as_u64()
            .unwrap_or(0)
            > 0,
        "the flash leg must actually dispatch Fused on this padded, head_dim=64, bf16 fixture — \
         tier={flash_tier}"
    );

    let block_report = run_report(padded_command(&dir).env(
        "JAMMI_KERNELS_DISABLE",
        // The SHIPPING explicit op-name form — NEVER `=all`, which would
        // also disable ops this A/B does not compare.
        "attention_block_flash,adamw_step_fused",
    ));
    let block_tier = &block_report["tiers"]["finetune_step"];
    assert_eq!(
        block_tier["attention_block_flash_fused_dispatches"].as_u64(),
        Some(0),
        "tier={block_tier}"
    );
    assert_eq!(
        block_tier["kernels_disabled_fired"],
        serde_json::json!(["adamw_step_fused", "attention_block_flash"]),
        "both named ops must have actually fired on this fixture, or the run is INVALID; \
         tier={block_tier}"
    );

    // Both legs ran the IDENTICAL padded fixture/seed — same identity,
    // same step count — so their loss sequences are structurally
    // comparable in SHAPE (never asserted near-equal: see note (a) above,
    // and the batched-vs-unbatched sibling test's own reasoning for why
    // two different arithmetic compositions are not expected to agree
    // bit-for-bit even when both are correct).
    assert_eq!(flash_tier["row_lengths"], block_tier["row_lengths"]);
    assert_eq!(flash_tier["seed"], block_tier["seed"]);
    let flash_losses = flash_tier["losses"].as_array().expect("flash losses");
    let block_losses = block_tier["losses"].as_array().expect("block losses");
    assert_eq!(flash_losses.len(), block_losses.len());
    for loss in flash_losses.iter().chain(block_losses.iter()) {
        assert!(
            loss.as_f64().expect("loss is a number").is_finite(),
            "every recorded loss on both arms must be finite — flash={flash_losses:?} \
             block={block_losses:?}"
        );
    }
}
