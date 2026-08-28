//! CPU-hermetic end-to-end smoke test for the `finetune-run` tier (unit 63,
//! CONTRACT H4): drives the REAL compiled `jammi-bench finetune-run`
//! subcommand — never `finetune_run::run` in-process (this crate is
//! `[[bin]]`-only, see `finetune_step_kernel_disable.rs`'s own doc for why a
//! fresh child process is this crate's convention) — over a TINY generic
//! fixture (`jammi-test-utils`' committed `tiny_bert`, BERT architecture,
//! real tokenizer) and a hand-written 2-batch synthetic triplet set, proving
//! this tier actually drives `TrainingLoopBuilder` + the public
//! `evaluate_held_out` seam end to end and emits a well-formed report with
//! every identity field non-null.
//!
//! `--epochs 2 --eval-cadence 1` deliberately exercises this tier's
//! resume-cycle TWICE (not just once), so a smoke run that only worked for a
//! single fresh (non-resumed) epoch would still fail here.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `cookbook/fixtures/tiny_bert` — the SAME generic, committed fixture
/// `jammi_test_utils::cookbook_fixture("tiny_bert")` resolves to
/// (`workspace_root().join("cookbook").join("fixtures")`), spelled as a
/// relative path here (mirroring `finetune_step_kernel_disable.rs`'s own
/// `model_dir()`) rather than adding `jammi-test-utils` as a dev-dependency
/// of this `[[bin]]`-only crate — BERT architecture, real `tokenizer.json`,
/// no `1_Pooling/` (falls back to mean pooling), no consumer shape.
fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

/// Write `n` synthetic (anchor, positive, negative) triplets as JSONL, using
/// the SAME field names the committed `finetune_heldout` fixture (CONTRACT
/// H3) uses, so this generic synthetic fixture and the real committed one
/// are structurally interchangeable inputs to this CLI.
fn write_triplets_jsonl(dir: &Path, name: &str, n: usize, offset: usize) -> PathBuf {
    let path = dir.join(name);
    let mut body = String::new();
    for i in 0..n {
        let k = offset + i;
        body.push_str(&format!(
            "{{\"anchor_id\":\"a{k}\",\"anchor_text\":\"synthetic anchor sentence number {k} \
             about widgets\",\"positive_id\":\"p{k}\",\"positive_text\":\"synthetic positive \
             sentence number {k} about widgets too\",\"negative_id\":\"n{k}\",\"negative_text\":\
             \"synthetic negative sentence number {k} about gadgets instead\"}}\n"
        ));
    }
    std::fs::write(&path, body).expect("write triplets jsonl");
    path
}

/// Write the committed held-out id-order file:
/// `anchor_id\tpositive_id\tnegative_id` per line, in COMMITTED order.
fn write_heldout_ids(dir: &Path, n: usize, offset: usize) -> PathBuf {
    let path = dir.join("heldout_ids.txt");
    let mut body = String::new();
    for i in 0..n {
        let k = offset + i;
        body.push_str(&format!("a{k}\tp{k}\tn{k}\n"));
    }
    std::fs::write(&path, body).expect("write heldout ids");
    path
}

/// One `finetune-run` invocation over the tiny synthetic fixture: 4 train
/// triplets (2 batches at `--batch 2`), 2 held-out triplets (1 batch),
/// 2 epochs, `--eval-cadence 1` (so both epochs call `evaluate_held_out`).
fn base_command(work_dir: &Path, fixtures_dir: &Path) -> Command {
    let train_jsonl = write_triplets_jsonl(fixtures_dir, "train.jsonl", 4, 0);
    let heldout_jsonl = write_triplets_jsonl(fixtures_dir, "heldout.jsonl", 2, 100);
    let heldout_ids = write_heldout_ids(fixtures_dir, 2, 100);

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args(["finetune-run", "--model-dir"])
        .arg(model_dir())
        .args(["--arm", "fused"])
        .arg("--train-jsonl")
        .arg(&train_jsonl)
        .arg("--heldout-ids")
        .arg(&heldout_ids)
        .arg("--heldout-jsonl")
        .arg(&heldout_jsonl)
        .args([
            "--seed",
            "7",
            "--epochs",
            "2",
            "--eval-cadence",
            "1",
            "--batch",
            "2",
            "--lr",
            "0.001",
            "--schedule",
            "constant",
            "--warmup-steps",
            "0",
            "--weight-decay",
            "0.0",
            "--grad-accum",
            "1",
            "--validation-fraction",
            "0.0",
            "--early-stopping-patience",
            "10000",
            "--early-stopping-metric",
            "train_loss",
            "--max-grad-norm",
            "0.0",
            "--margin",
            "0.3",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0.0",
            "--target-modules",
            "query,value",
            "--backbone-dtype",
            "f32",
            "--max-seq-length",
            "16",
        ])
        .arg("--work-dir")
        .arg(work_dir);
    cmd
}

#[test]
fn finetune_run_smoke_end_to_end_cpu_hermetic() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path())
        .output()
        .expect("spawn jammi-bench finetune-run");
    assert!(
        output.status.success(),
        "finetune-run exited non-zero: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse finetune-run report JSON");
    let tier = report
        .get("tiers")
        .and_then(|t| t.get("finetune_run"))
        .expect("report.tiers.finetune_run present");
    let obj = tier.as_object().expect("finetune_run tier is an object");

    // Every field this tier declares IDENTITY or PROVENANCE must be
    // present, and every NonNull-declared one must not be `null` — the
    // same mechanical check `finetune_run::run` itself already performs
    // before returning, re-checked here against the REAL CLI's stdout.
    for field in [
        "seed",
        "batch",
        "seq",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "margin",
        "target_modules",
        "batched_forward",
        "backbone_dtype",
        "steps_measured",
        "checkpoint_config_sha256",
        "checkpoint_weights_sha256",
        "checkpoint_weights_size_bytes",
        "epochs",
        "lr",
        "schedule",
        "warmup_steps",
        "weight_decay",
        "grad_accum",
        "validation_fraction",
        "split_rule",
        "split_seed",
        "dataset_sha256",
        "heldout_ids_sha256",
        "heldout_batch_partition_sha256",
        "embedding_loss",
        "matryoshka_dims",
        "early_stopping_patience",
        "early_stopping_metric",
        "eval_cadence",
    ] {
        let v = obj
            .get(field)
            .unwrap_or_else(|| panic!("finetune_run tier missing identity field {field:?}"));
        assert!(!v.is_null(), "identity field {field:?} is null: {v:?}");
    }
    for field in [
        "arm",
        "device_name",
        "kernels_disabled_requested",
        "kernels_disabled_fired",
        "flash_compiled",
        "build_features",
        "attention_arm",
    ] {
        let v = obj
            .get(field)
            .unwrap_or_else(|| panic!("finetune_run tier missing provenance field {field:?}"));
        assert!(!v.is_null(), "provenance field {field:?} is null: {v:?}");
    }

    // The endpoint fields (CONTRACT H4/Frame).
    assert_eq!(obj["final_epoch"], serde_json::json!(1));
    assert!(obj["held_out_example_mean"].as_f64().is_some());
    assert_eq!(obj["held_out_count"], serde_json::json!(2));
    assert!(obj["final_loss_diagnostic"].as_f64().is_some());

    // `--eval-cadence 1` over 2 epochs must produce a two-point trajectory —
    // proof the resume-cycle actually ran BOTH epochs (not just a single
    // fresh, non-resumed one).
    let trajectory = obj["trajectory"].as_array().expect("trajectory array");
    assert_eq!(
        trajectory.len(),
        2,
        "expected one evaluate_held_out point per epoch at eval_cadence=1: {trajectory:?}"
    );
    assert_eq!(trajectory[0]["epoch"], serde_json::json!(0));
    assert_eq!(trajectory[1]["epoch"], serde_json::json!(1));

    // `--arm fused` was declared; the process made no kernel-disable claim.
    assert_eq!(obj["arm"], serde_json::json!("fused"));
}
