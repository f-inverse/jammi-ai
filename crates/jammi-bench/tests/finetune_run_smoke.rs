//! CPU-hermetic end-to-end smoke test for the `finetune-run` tier: drives
//! the REAL compiled `jammi-bench finetune-run`
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
/// the SAME field names the committed `finetune_heldout` fixture uses, so
/// this generic synthetic fixture and the real committed one
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
/// `objective` is `"triplet"` or `"mnrl"`: the SAME fixture rows and
/// `--heldout-ids` order
/// feed either — `mnrl` drops the negative column via the tier's own
/// `project_to_pairs` projection.
fn base_command(work_dir: &Path, fixtures_dir: &Path, objective: &str) -> Command {
    base_command_with_epochs(work_dir, fixtures_dir, objective, 2)
}

/// [`base_command`] with the epoch count as a parameter. `2` is the
/// resume-cycle case most tests here drive; `1` is the shape the profile
/// legs pin, which
/// [`fusible_site_census_satisfies_the_positive_proof_equation_on_a_real_run`]
/// runs at.
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
        "max_seq_length",
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

    // The endpoint fields, and the origin the endpoint is measured from.
    assert!(obj["held_out_at_init"].as_f64().is_some());
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

    // `train_probe_series` must exist
    // and carry exactly `epochs + 1` entries (the init probe plus one per
    // epoch) in the CLI-level report the
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

    // GELU-erf positive-proof: `tiny_bert`'s
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

    // Identity-value semantics: `--objective triplet` → `embedding_loss: "triplet"`,
    // `temperature: null`, `margin` non-null (already checked above).
    assert_eq!(obj["embedding_loss"], serde_json::json!("triplet"));
    assert!(
        obj["temperature"].is_null(),
        "Triplet run must report temperature: null, got {:?}",
        obj["temperature"]
    );
}

/// The profile's POSITIVE-PROOF equation, checked LIVE on a real run: for
/// each fusible key, `fused + eager ==
/// <witnessed census field> × steps_measured`.
///
/// This is the assertion a profile merger applies to
/// every pod leg, run here against the CLI's own stdout so the equation is
/// proven to HOLD on real production output rather than only on
/// synthetic JSON. A subprocess is
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
/// ## The `--grad-accum 1` pin is LOAD-BEARING
///
/// `steps_measured` counts OPTIMIZER steps, and the equation's `batches`
/// term is training FORWARDS; the two coincide only at `--grad-accum 1`,
/// which is what the profile legs pin and what this test asserts before it
/// uses one for the other. The epoch count does not enter: the tier reads the
/// trainer's absolute step counter off the final resume leg, so a
/// multi-epoch run counts each step once
/// (`finetune_run_emits_a_reproducible_pairing_surface` pins that at
/// `--epochs 2`). This test runs at `--epochs 1`, the profile legs' own
/// shape.
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
    // forward), at the profile legs' own `--epochs 1`. Eval
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
        "4 train rows at --batch 2 over one epoch is 2 optimizer steps"
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

/// The MNRL twin of [`finetune_run_smoke_end_to_end_cpu_hermetic`]: the
/// SAME fixture, SAME held-out id order, `--objective mnrl`
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
        "max_seq_length",
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

    // Identity-value semantics: MNRL flips the nullness pair —
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

    // The MNRL objective's
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

/// Run the MNRL fixture and return the parsed `tiers.finetune_run` object
/// alongside the untrained adapter the run wrote into its work dir.
fn run_mnrl_in(scratch: &Path) -> (serde_json::Value, PathBuf) {
    let work_dir = scratch.join("work");
    let fixtures_dir = scratch.join("fixtures");
    std::fs::create_dir_all(&work_dir).expect("work dir");
    std::fs::create_dir_all(&fixtures_dir).expect("fixtures dir");
    let output = base_command(&work_dir, &fixtures_dir, "mnrl")
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
    (
        report["tiers"]["finetune_run"].clone(),
        work_dir.join("initial_adapter.safetensors"),
    )
}

/// The surface a run in another framework pairs against: the untrained
/// adapter it can load, the digest that proves both sides loaded the same
/// bytes, the realized token batches, and a time axis under the held-out
/// trajectory. Two separate processes at the same seed must agree on every
/// one of the digests — the init is a function of `(seed, parameter name)`
/// and tokenization of the corpus alone — or the pairing premise is false.
#[test]
fn finetune_run_emits_a_reproducible_pairing_surface() {
    use sha2::{Digest, Sha256};

    let first_scratch = tempfile::tempdir().expect("tempdir");
    let second_scratch = tempfile::tempdir().expect("tempdir");
    let (first, first_adapter) = run_mnrl_in(first_scratch.path());
    let (second, second_adapter) = run_mnrl_in(second_scratch.path());

    // The recorded digest is the digest of the file on disk.
    let adapter_bytes = std::fs::read(&first_adapter).expect("read initial adapter");
    assert_eq!(
        first["initial_adapter_sha256"],
        serde_json::json!(hex::encode(Sha256::digest(&adapter_bytes))),
        "initial_adapter_sha256 must digest the adapter file the run wrote"
    );
    assert_eq!(
        adapter_bytes,
        std::fs::read(&second_adapter).expect("read second initial adapter"),
        "two processes at one seed must dump byte-identical untrained adapters"
    );

    for field in ["train_token_ids_sha256", "heldout_token_ids_sha256"] {
        let digest = first[field]
            .as_str()
            .unwrap_or_else(|| panic!("{field} must be a hex digest on a text run"));
        assert_eq!(digest.len(), 64, "{field} is not a sha256 hex: {digest:?}");
        assert_eq!(
            first[field], second[field],
            "{field} must be reproducible across processes"
        );
    }
    assert_ne!(
        first["train_token_ids_sha256"], first["heldout_token_ids_sha256"],
        "the train and held-out token streams are different corpora"
    );

    // Each optimizer step is counted once however many resume legs the run
    // spans: 4 train rows at `--batch 2 --grad-accum 1` is 2 steps per epoch.
    assert_eq!(
        first["steps_measured"],
        serde_json::json!(4),
        "2 epochs x 2 steps; the resumed second leg must not re-count the first leg's steps"
    );

    // The time axis: one wall per epoch leg, whole and by phase. The legs'
    // walls sum to the total; each leg's phases are disjoint spans inside it;
    // and each trajectory point carries both running sums at its epoch's end.
    // This fixture monitors `train_loss`, so no leg has a validation wall.
    let walls = first["epoch_walls"].as_array().expect("epoch_walls array");
    assert_eq!(walls.len(), 2, "one wall per epoch leg at --epochs 2");
    let seconds = |wall: &serde_json::Value, field: &str| {
        wall[field]
            .as_f64()
            .unwrap_or_else(|| panic!("epoch wall {field} is not a number: {wall:?}"))
    };
    for wall in walls {
        assert!(seconds(wall, "steps_s") > 0.0, "every leg trains: {wall:?}");
        assert!(
            seconds(wall, "checkpoint_s") > 0.0,
            "every leg checkpoints: {wall:?}"
        );
        assert_eq!(seconds(wall, "validation_s"), 0.0, "{wall:?}");
        assert!(
            seconds(wall, "steps_s") + seconds(wall, "checkpoint_s") <= seconds(wall, "run_s"),
            "a leg's phases are spans inside its run() call: {wall:?}"
        );
    }
    let total = first["train_run_wall_s"]
        .as_f64()
        .expect("train_run_wall_s");
    assert_eq!(
        walls.iter().map(|w| seconds(w, "run_s")).sum::<f64>(),
        total
    );
    let trajectory = first["trajectory"].as_array().expect("trajectory array");
    assert_eq!(
        trajectory[0]["run_wall_s_cumulative"].as_f64(),
        Some(seconds(&walls[0], "run_s"))
    );
    assert_eq!(trajectory[1]["run_wall_s_cumulative"].as_f64(), Some(total));
    assert_eq!(
        trajectory[1]["steps_wall_s_cumulative"].as_f64(),
        Some(walls.iter().map(|w| seconds(w, "steps_s")).sum::<f64>())
    );

    // Host memory is the kernel's high-water mark wherever the kernel
    // exposes one; device memory is unmeasured on a host with no device
    // probe — absent, never a fabricated zero.
    assert_eq!(first["peak_rss_bytes"]["unit"], serde_json::json!("bytes"));
    if cfg!(target_os = "linux") {
        assert!(
            first["peak_rss_bytes"]["value"].as_f64().unwrap_or(0.0) > 0.0,
            "VmHWM must be measured on Linux: {:?}",
            first["peak_rss_bytes"]
        );
    }
    assert_eq!(first["peak_vram_bytes"]["unit"], serde_json::json!("bytes"));
}

/// The negative control through the real CLI: `--zero-lr-control` beside the
/// job's own positive `--lr`. The leg runs to completion, reports `lr: 0.0`,
/// and its train probe never moves, so the learning-happened delta a reader
/// derives (`series[0] - series[last]`) is exactly `0.0`. `--lr 0`, which is
/// not a job anyone can submit, is refused and names the flag.
#[test]
fn a_zero_lr_control_leg_runs_and_reports_no_learning() {
    let work_dir = tempfile::tempdir().expect("tempdir");
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let output = base_command(work_dir.path(), fixtures_dir.path(), "mnrl")
        .arg("--zero-lr-control")
        .output()
        .expect("spawn jammi-bench finetune-run");
    assert!(
        output.status.success(),
        "a control leg must run: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse finetune-run report JSON");
    let tier = &report["tiers"]["finetune_run"];
    assert_eq!(tier["lr"], serde_json::json!(0.0));
    assert_eq!(tier["steps_measured"], serde_json::json!(4));
    let series: Vec<f64> = tier["train_probe_series"]
        .as_array()
        .expect("train_probe_series array")
        .iter()
        .map(|p| p.as_f64().expect("probe is a number"))
        .collect();
    assert_eq!(series.len(), 3);
    assert_eq!(
        series[0] - series[series.len() - 1],
        0.0,
        "learning_happened_delta of a control leg must be exactly zero: {series:?}"
    );

    // The same command with its `--lr` value replaced by `0`.
    let refused_work_dir = tempfile::tempdir().expect("tempdir");
    let template = base_command(refused_work_dir.path(), fixtures_dir.path(), "mnrl");
    let mut args: Vec<std::ffi::OsString> = template.get_args().map(Into::into).collect();
    let lr_value = args
        .iter()
        .position(|arg| arg == "--lr")
        .expect("base_command passes --lr")
        + 1;
    args[lr_value] = "0".into();
    let refused = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(args)
        .output()
        .expect("spawn jammi-bench finetune-run");
    assert!(!refused.status.success(), "--lr 0 must be refused");
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("--zero-lr-control"),
        "the refusal must name the control: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
}
