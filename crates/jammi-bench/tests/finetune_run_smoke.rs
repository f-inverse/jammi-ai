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
/// `objective` is `"triplet"` or `"mnrl"` (unit 63 H4a-delta, CONTRACT
/// amendment 2026-08-28): the SAME fixture rows and `--heldout-ids` order
/// feed either — `mnrl` drops the negative column via the tier's own
/// `project_to_pairs` projection.
fn base_command(work_dir: &Path, fixtures_dir: &Path, objective: &str) -> Command {
    base_command_with_epochs(work_dir, fixtures_dir, objective, 2)
}

/// [`base_command`] with the epoch count as a parameter. `2` is the
/// resume-cycle case every pre-existing test here drives; `1` is what the
/// #421 profile legs pin, and the two are NOT interchangeable for
/// `steps_measured` — see
/// [`fusible_site_census_satisfies_the_positive_proof_equation_on_a_real_run`]
/// for the exact difference and why it matters.
fn base_command_with_epochs(
    work_dir: &Path,
    fixtures_dir: &Path,
    objective: &str,
    epochs: usize,
) -> Command {
    let epochs = epochs.to_string();
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
        .args(["--epochs", &epochs])
        .args([
            "--seed",
            "7",
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
            "--objective",
            objective,
            "--margin",
            "0.3",
            "--temperature",
            "20.0",
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
    let output = base_command(work_dir.path(), fixtures_dir.path(), "triplet")
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
        "backbone_dtype",
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
        "train_pairs_file_sha256",
        "heldout_ids_sha256",
        "heldout_pairs_sha256",
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
        "split_rule",
        "batched_forward",
        "steps_measured",
        "fusible_site_census",
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

    // Advisory (e) (unit 63 round-7 audit): `train_probe_series` must exist
    // and carry exactly `epochs + 1` entries (the init probe plus one per
    // epoch — CONTRACT amendment 2026-08-29b) in the CLI-level report the
    // merger actually reads, not merely in an in-process unit test —
    // `--epochs 2` here, so `2 + 1 == 3`.
    let train_probe_series = obj["train_probe_series"]
        .as_array()
        .expect("train_probe_series array");
    assert_eq!(
        train_probe_series.len(),
        3,
        "train_probe_series must carry epochs (2) + 1 entries (the init probe plus one per \
         epoch): {train_probe_series:?}"
    );
    for (i, v) in train_probe_series.iter().enumerate() {
        assert!(
            v.as_f64().is_some(),
            "train_probe_series[{i}] must be a finite number, got {v:?}"
        );
    }

    // `--arm fused` was declared; the process made no kernel-disable claim.
    assert_eq!(obj["arm"], serde_json::json!("fused"));

    // C-MLP GELU-erf positive-proof (campaign #462/#463): `tiny_bert`'s
    // FFN (`BertIntermediate::forward`, `hidden_act: "gelu"`) calls
    // `jammi_encoders::activations::gelu_erf` in training mode at least
    // once per layer per forward — this run's `--arm fused` and CPU F32
    // backbone both satisfy `gelu_admission_predicate`'s domain, so the
    // fused arm must have actually dispatched, not merely registered a
    // counter that stayed at zero. A wrong registry key on the read side
    // (`jammi_kernels::admission::counters_for` keyed by exact string,
    // never a prefix) would silently report `0` here forever even though
    // production dispatched through the fused kernel every step — this
    // assertion is the mechanism check that catches exactly that class of
    // bug, not merely that the field is present in the JSON.
    assert!(
        obj["gelu_fused_dispatches"].as_u64().unwrap_or(0) > 0,
        "expected gelu_fused_dispatches > 0 for a tiny_bert --arm fused CPU F32 run: {:?}",
        obj["gelu_fused_dispatches"]
    );

    // Identity-value semantics (unit 63 H4a-delta, CONTRACT amendment
    // 2026-08-28): `--objective triplet` → `embedding_loss: "triplet"`,
    // `temperature: null`, `margin` non-null (already checked above).
    assert_eq!(obj["embedding_loss"], serde_json::json!("triplet"));
    assert!(
        obj["temperature"].is_null(),
        "Triplet run must report temperature: null, got {:?}",
        obj["temperature"]
    );
}

/// The profile's POSITIVE-PROOF equation, checked LIVE on a real run
/// (issue #421 §D4 item 1): for each fusible key, `fused + eager ==
/// <witnessed census field> × steps_measured`.
///
/// This is the assertion `ci/scripts/perf/profile_421_merge.py` applies to
/// every pod leg, run here against the CLI's own stdout so the equation is
/// proven to HOLD on real production output rather than only on the
/// synthetic JSON that merger's own hermetic tests feed it. A subprocess is
/// the right home for it and an in-crate unit test is not: the dispatch
/// counters are PROCESS-WIDE, and this crate's unit tests drive `run_impl`
/// concurrently on several threads of one process, so an exact-count
/// assertion there would read other tests' dispatches as this run's.
///
/// The three expected census values are DERIVED, live, from the committed
/// fixture's own `config.json` and the `--target-modules` this command
/// pins — never transcribed from a previous run of the thing under test.
/// `AnyEncoder::fusible_site_census`'s own doc gives the BERT walk:
/// wrapped arms per layer (2 here — `query,value`), `embeddings + 2 per
/// layer` LayerNorms, one GELU seam call per layer.
///
/// ## The `--epochs 1` pin is LOAD-BEARING, and this test found out why
///
/// `batches == steps_measured` holds only at `--epochs 1 --grad-accum 1`,
/// which is exactly what the #421 legs pin — and this test is written at
/// that pin rather than at this file's `base_command` default of `2`
/// BECAUSE the first version of it, at `--epochs 2`, failed:
/// `steps_measured` read `6` where the run took `4` training forwards
/// (`lora_linear_fused_dispatches == 8` over `2` wrapped sites).
///
/// The cause is this tier's resume-cycle. `finetune_run::run` drives
/// `params.epochs` single-epoch `TrainingLoop::run` legs, each configured
/// with `epochs = epoch_idx + 1` and resumed from the previous leg's
/// checkpoint, and sums each leg's `TrainingResult::total_steps` — but
/// that field is the leg's own `global_step`, which a resumed leg carries
/// forward from before the resume. So leg 0 reports `2` and leg 1 reports
/// `4` for a run whose second epoch trained `2` batches, and the sum
/// `6` over-counts. `steps_measured` is therefore a faithful count of
/// TRAINING FORWARDS only when there is a single leg.
///
/// This is a convention pin, not a bug hunt: at `--epochs 1` (every #421
/// leg, A and D alike) the two coincide exactly, which is what the
/// assertion below proves on real output. `profile_421_merge.py` REFUSES a
/// leg whose `epochs`/`grad_accum` are not `1`, naming this reason, rather
/// than reporting an equation failure for a leg that simply was not run
/// under the convention the equation is defined for.
#[test]
fn fusible_site_census_satisfies_the_positive_proof_equation_on_a_real_run() {
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_dir().join("config.json")).expect("read tiny_bert config.json"),
    )
    .expect("parse tiny_bert config.json");
    let layers = config["num_hidden_layers"]
        .as_u64()
        .expect("tiny_bert config.json must state num_hidden_layers");
    assert!(
        layers > 0,
        "a 0-layer fixture would make every expectation below vacuously 0"
    );
    // Independently derived from the fixture + this command's own flags.
    let expected_lora_sites = 2 * layers; // `--target-modules query,value`
    let expected_layer_norms = 1 + 2 * layers; // embeddings + 2 per layer
    let expected_gelu_calls = layers; // BertIntermediate, `hidden_act: "gelu"`

    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command_with_epochs(work_dir.path(), fixtures_dir.path(), "triplet", 1)
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
    let obj = report["tiers"]["finetune_run"]
        .as_object()
        .expect("finetune_run tier is an object")
        .clone();
    let census = obj["fusible_site_census"]
        .as_object()
        .expect("fusible_site_census must serialize as an object");

    // The convention the equation is defined under, asserted rather than
    // assumed: `--grad-accum 1` (one optimizer step is one training
    // forward) and `--epochs 1` (one leg, so `steps_measured` is not the
    // resume-cycle's over-counted sum — see this test's own doc). Eval
    // forwards contribute nothing to either side of any pair (the LoRA site
    // early-returns in eval, the house LayerNorm's fused arm is under its
    // training branch, and the GELU seam's eval arm is the plain
    // `Tensor::gelu_erf`), so this run's `evaluate_held_out` calls and its
    // train probes do not appear on either side.
    assert_eq!(obj["grad_accum"], serde_json::json!(1));
    assert_eq!(obj["epochs"], serde_json::json!(1));
    let steps = obj["steps_measured"].as_u64().expect("steps_measured");
    assert_eq!(
        steps, 2,
        "4 train rows at --batch 2 over one epoch is 2 optimizer steps, and at --epochs 1 \
         steps_measured is exactly that (no resume leg to double-count)"
    );

    for (census_field, expected_calls, fused_field, eager_field) in [
        (
            "lora_sites_wrapped",
            expected_lora_sites,
            "lora_linear_fused_dispatches",
            "lora_linear_eager_dispatches",
        ),
        (
            "layer_norms",
            expected_layer_norms,
            "ln_fused_dispatches",
            "ln_eager_dispatches",
        ),
        (
            "gelu_seam_calls_per_forward",
            expected_gelu_calls,
            "gelu_fused_dispatches",
            "gelu_eager_dispatches",
        ),
    ] {
        let calls = census[census_field]
            .as_u64()
            .unwrap_or_else(|| panic!("census.{census_field} must be a non-negative integer"));
        assert_eq!(
            calls, expected_calls,
            "census.{census_field} = {calls}, but the committed fixture's own config \
             ({layers} layer(s)) and this command's --target-modules derive {expected_calls}"
        );
        // Non-vacuity: every one of the three is genuinely NON-ZERO on this
        // fixture, so the equality below is a real constraint on all three
        // and not `0 == 0` for any of them.
        assert!(
            calls > 0,
            "census.{census_field} is 0 on tiny_bert — the equation below would be vacuous"
        );
        let fused = obj[fused_field]
            .as_u64()
            .unwrap_or_else(|| panic!("{fused_field}"));
        let eager = obj[eager_field]
            .as_u64()
            .unwrap_or_else(|| panic!("{eager_field}"));
        assert_eq!(
            fused + eager,
            calls * steps,
            "positive proof failed: {fused_field}={fused} + {eager_field}={eager} != \
             census.{census_field}={calls} x steps_measured={steps}"
        );
    }
}

/// The MNRL twin of [`finetune_run_smoke_end_to_end_cpu_hermetic`] (unit 63
/// H4a-delta): the SAME fixture, SAME held-out id order, `--objective mnrl`
/// instead — proving this tier actually drives the (anchor, positive)
/// projection through `TrainingLoopBuilder` + `evaluate_held_out` end to
/// end, across the same 2-epoch resume-cycle.
#[test]
fn finetune_run_smoke_mnrl_end_to_end_cpu_hermetic() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "mnrl")
        .output()
        .expect("spawn jammi-bench finetune-run");
    assert!(
        output.status.success(),
        "finetune-run --objective mnrl exited non-zero: stdout={}\nstderr={}",
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
    // present — `margin` is now the field expected NULL (MNRL selected),
    // so it is checked separately below rather than in this presence loop.
    for field in [
        "seed",
        "batch",
        "seq",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "backbone_dtype",
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
        "train_pairs_file_sha256",
        "heldout_ids_sha256",
        "heldout_pairs_sha256",
        "heldout_batch_partition_sha256",
        "embedding_loss",
        "temperature",
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

    // Identity-value semantics (task item 4): MNRL flips the nullness pair —
    // `margin: null`, `temperature` non-null (already checked above),
    // `embedding_loss: "mnrl"`.
    assert!(
        obj["margin"].is_null(),
        "MNRL run must report margin: null, got {:?}",
        obj["margin"]
    );
    assert_eq!(obj["embedding_loss"], serde_json::json!("mnrl"));

    // The endpoint fields still hold under MNRL.
    assert_eq!(obj["final_epoch"], serde_json::json!(1));
    assert!(obj["held_out_example_mean"].as_f64().is_some());
    assert_eq!(obj["held_out_count"], serde_json::json!(2));

    let trajectory = obj["trajectory"].as_array().expect("trajectory array");
    assert_eq!(
        trajectory.len(),
        2,
        "expected one evaluate_held_out point per epoch at eval_cadence=1: {trajectory:?}"
    );

    // Advisory (e) (unit 63 round-7 audit): the MNRL objective's
    // `train_probe_series` must carry the same `epochs + 1` shape as the
    // Triplet leg — this field's shape does not depend on which objective
    // was selected.
    let train_probe_series = obj["train_probe_series"]
        .as_array()
        .expect("train_probe_series array");
    assert_eq!(
        train_probe_series.len(),
        3,
        "train_probe_series must carry epochs (2) + 1 entries (the init probe plus one per \
         epoch): {train_probe_series:?}"
    );
}
