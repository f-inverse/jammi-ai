//! CPU-hermetic end-to-end smoke test for `finetune-run --task
//! image_embedding` / `--task audio_embedding` (issue #421 W2b): drives the
//! REAL compiled `jammi-bench finetune-run` subcommand over the COMMITTED
//! media producers' output and the committed tiny OpenCLIP / HF-CLAP
//! fixtures, proving the whole chain — producer → media JSONL → media row
//! loader → `Checkpoint::resolve` → the tower builder → `TrainingLoopBuilder`
//! with the run's task → `evaluate_held_out` — runs and emits a well-formed
//! report.
//!
//! # Why the corpus comes from the Python producers
//!
//! The contract offered two ways to get media fixtures into this test: call
//! `ci/scripts/perf/gen_fixed_shape_image_corpus.py` when `python3` is
//! present, or synthesise the files in Rust. This file takes the FIRST,
//! deliberately: those producers ARE PR B's declared workload, and a test
//! that generated its own lookalike files would prove the loader reads
//! *something* while leaving the actual profile inputs unexercised. The
//! producers are stdlib-only and offline, so the choice costs no hermeticity
//! — only a `python3` on PATH, which every lane that runs
//! `ci/scripts/perf/test_*.py` already has.
//!
//! When `python3` is genuinely unavailable the test prints an explicit
//! stderr warning and returns rather than asserting — a silent pass would be
//! worse than a stated non-run, and the in-crate unit tests
//! (`finetune_run::tests::build_encoder_adapters_builds_the_*_tower`) cover
//! the tower dispatch on the committed fixtures with no Python at all.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The FULL per-tower LoRA site sets `ci/scripts/perf/profile_421_legs.sh`
/// pins on every real pod leg (`CLIP_FULL`/`CLAP_FULL` there) — spelled out
/// literally here, never imported (this crate is `[[bin]]`-only, mirroring
/// every other test file in this directory's own convention of re-deriving
/// fixture/constant values locally; see `finetune_run_kernel_disable.rs`'s
/// own doc for the same reasoning about `ALLOFF_KEYS`).
///
/// Unit-467 pressure-test folded advisory: the positive-proof equation
/// assertions below (`assert_positive_proof_equation`) are witnessed at
/// THESE site sets specifically, not the earlier `in_proj,c_fc`/
/// `query,value` this file used before — matching the REAL profile leg's
/// shape is what makes this smoke test's witnessed census numbers directly
/// comparable to a pod leg's, rather than a shape only this file has ever
/// exercised.
const CLIP_FULL_TARGET_MODULES: &str = "in_proj,out_proj,c_fc,c_proj";
const CLAP_FULL_TARGET_MODULES: &str =
    "query,key,value,attention_output,intermediate_dense,output_dense,reduction,linear1,linear2";

/// KO-7 require-gate helper for every `python3`-unavailable skip below (to
/// be registered in `ci/kernel-oracle-helpers.txt`): a lane that
/// specifically wants to prove the media end-to-end legs (the hermetic CI
/// runner, which has `python3`) sets `JAMMI_REQUIRE_MEDIA_SMOKE` — if that
/// lane's box unexpectedly cannot launch `python3` (so the producer, and
/// therefore the whole leg, cannot be observed), this is a hard failure,
/// never a silent skip.
fn media_producer_require_gate() {
    if std::env::var_os("JAMMI_REQUIRE_MEDIA_SMOKE").is_some() {
        panic!(
            "finetune_run_media_smoke: python3 is not runnable but JAMMI_REQUIRE_MEDIA_SMOKE is set; a silent skip is not acceptable"
        );
    }
}

/// Run one committed producer into `out_dir`. Returns `false` (having said
/// so on stderr) when `python3` cannot be launched at all; panics when the
/// producer itself fails, which is a real regression, not an environment gap.
fn run_producer(script: &str, out_dir: &Path, rows: usize, extra: &[&str]) -> bool {
    let mut cmd = Command::new("python3");
    cmd.current_dir(repo_root())
        .arg(script)
        .arg("--rows")
        .arg(rows.to_string())
        .arg("--seed")
        .arg("5")
        .arg("--out-dir")
        .arg(out_dir)
        .args(extra);
    match cmd.output() {
        Ok(output) => {
            assert!(
                output.status.success(),
                "{script} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            true
        }
        Err(e) => {
            eprintln!(
                "python3 is not runnable here ({e}); the media end-to-end leg for {script} was \
                 NOT exercised in this run"
            );
            false
        }
    }
}

/// Write the committed held-out id-order file from the first `n` rows of the
/// producer's own JSONL — the scoring order is the ids file's, so this also
/// exercises the media held-out join.
fn write_heldout_ids(corpus: &Path, n: usize) -> PathBuf {
    let jsonl = corpus.join("triplets.jsonl");
    let text = std::fs::read_to_string(&jsonl).expect("read producer jsonl");
    let mut body = String::new();
    for line in text.lines().take(n) {
        let row: serde_json::Value = serde_json::from_str(line).expect("parse producer row");
        body.push_str(&format!(
            "{}\t{}\t{}\n",
            row["anchor_id"].as_str().expect("anchor_id"),
            row["positive_id"].as_str().expect("positive_id"),
            row["negative_id"].as_str().expect("negative_id"),
        ));
    }
    let path = corpus.join("heldout_ids.txt");
    std::fs::write(&path, body).expect("write heldout ids");
    path
}

/// One `finetune-run` invocation over a media corpus. Deliberately mirrors
/// `finetune_run_smoke.rs`'s flag set so the two differ only in `--task`,
/// `--model-dir`, `--target-modules` and `--objective`.
///
/// `--objective` is a PARAMETER rather than a pinned literal a caller could
/// append a second copy of: `clap` refuses a repeated non-multiple argument
/// outright ("cannot be used multiple times"), so an override-by-appending
/// would have tested clap's own arity checking instead of this tier's
/// media/objective refusal.
fn media_command(
    model_dir: &Path,
    task: &str,
    corpus: &Path,
    work_dir: &Path,
    target_modules: &str,
    objective: &str,
) -> Command {
    let heldout_ids = write_heldout_ids(corpus, 4);
    let jsonl = corpus.join("triplets.jsonl");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args(["finetune-run", "--model-dir"])
        .arg(model_dir)
        .args(["--arm", "fused", "--task", task])
        .arg("--train-jsonl")
        .arg(&jsonl)
        .arg("--heldout-ids")
        .arg(&heldout_ids)
        .arg("--heldout-jsonl")
        .arg(&jsonl)
        .args([
            "--seed",
            "7",
            "--epochs",
            "1",
            "--eval-cadence",
            "1",
            "--batch",
            "4",
            "--lr",
            "0.001",
            "--schedule",
            "constant",
            "--validation-fraction",
            "0.25",
            "--early-stopping-patience",
            "10000",
            "--early-stopping-metric",
            "train_loss",
            "--max-grad-norm",
            "0.0",
            "--objective",
            objective,
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            "0.0",
            "--target-modules",
            target_modules,
            "--backbone-dtype",
            "f32",
        ])
        .arg("--work-dir")
        .arg(work_dir);
    cmd
}

/// Parse the emitted report and assert the shape every media leg must carry.
fn assert_well_formed_media_report(stdout: &str, task: &str) {
    let report: serde_json::Value =
        serde_json::from_str(stdout).unwrap_or_else(|e| panic!("{task}: report must be JSON: {e}"));
    let tier = &report["tiers"]["finetune_run"];
    assert!(
        !tier.is_null(),
        "{task}: report must carry a finetune_run tier"
    );
    let series = tier["train_probe_series"]
        .as_array()
        .unwrap_or_else(|| panic!("{task}: train_probe_series must be an array"));
    // One init probe + one per epoch: the resume-cycle actually ran.
    assert_eq!(series.len(), 2, "{task}: {series:?}");
    for v in series {
        let x = v
            .as_f64()
            .unwrap_or_else(|| panic!("{task}: probe must be a number"));
        assert!(
            x.is_finite(),
            "{task}: a non-finite probe is a diverged run, not a datum ({x})"
        );
    }
    let trajectory = tier["trajectory"]
        .as_array()
        .unwrap_or_else(|| panic!("{task}: trajectory must be an array"));
    assert_eq!(trajectory.len(), 1, "{task}: {trajectory:?}");
    let mean = trajectory[0]["held_out_mean"]
        .as_f64()
        .unwrap_or_else(|| panic!("{task}: held_out_mean must be a number"));
    assert!(
        mean.is_finite(),
        "{task}: held_out_mean must be finite, got {mean}"
    );
    // The endpoint field a downstream merger reads, checked for FINITENESS
    // rather than merely for presence: `NaN > c` is false, so a threshold
    // check on a diverged media leg would silently pass.
    let endpoint = tier["held_out_example_mean"]
        .as_f64()
        .unwrap_or_else(|| panic!("{task}: held_out_example_mean must be a number"));
    assert!(
        endpoint.is_finite(),
        "{task}: held_out_example_mean must be finite, got {endpoint}"
    );
    // The digest is MEASURED off the files this run opened; a media leg that
    // silently resolved a different checkpoint would carry a different one.
    assert!(
        tier["checkpoint_weights_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64),
        "{task}: checkpoint_weights_sha256 must be a measured sha256"
    );
    // Issue #421 P1-b: the tower this leg trained is IDENTITY on the report,
    // not something a reader has to infer from the row shape. On a
    // multi-tower checkpoint (`tiny_open_clip` carries a text tower AND a
    // vision tower behind ONE `checkpoint_weights_sha256`) this is the only
    // field that says WHICH weights were trained.
    assert_eq!(
        tier["task"],
        serde_json::json!(task),
        "{task}: the tier must record the tower this run actually trained"
    );
    // `--objective triplet` is accepted, and RECORDED, for every one of the
    // three tasks this helper drives (the media refusal is `mnrl`-only —
    // see `a_media_task_under_the_mnrl_objective_is_refused` below).
    assert_eq!(
        tier["embedding_loss"],
        serde_json::json!("triplet"),
        "{task}: --objective triplet must be accepted and recorded for this task"
    );
    // Issue #421 P1-b: on a MEDIA task the JSONL is a manifest of PATHS, so
    // `train_pairs_file_sha256` cannot anchor the corpus CONTENT — the two
    // media digests do, and they must be real measured digests, not `null`.
    // On the text task they are `null` BY DESIGN (there the manifest IS the
    // content), which is the negative control that keeps the media
    // assertion from being satisfiable by a producer that stamps a constant
    // on every leg.
    let is_media = task != "text_embedding";
    for field in ["train_media_sha256", "heldout_media_sha256"] {
        if is_media {
            assert!(
                tier[field]
                    .as_str()
                    .is_some_and(|s| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())),
                "{task}: {field} must be a measured sha256 over the corpus content, got {}",
                tier[field]
            );
        } else {
            assert!(
                tier[field].is_null(),
                "{task}: {field} must be null on a text task (the manifest IS the content), \
                 got {}",
                tier[field]
            );
        }
    }
    // Issue #421 P1-b(v): the DIRECT media front-end timer. On a media leg
    // it must be a real, positive, FINITE measurement that is STRICTLY LESS
    // than the run's own training wall (it is measured inside
    // `TrainingLoop::run`, so it is a subset of that span) -- checking mere
    // presence would pass on a hardcoded 0.0, and checking `> 0.0` alone
    // would pass on a NaN-free-but-absurd value larger than the whole run.
    // On a TEXT leg it must be `null`, never `0.0`: the two mean different
    // things (see the field's own doc), and a producer stamping `0.0`
    // everywhere would satisfy a presence-only check on both.
    let train_wall = tier["train_run_wall_s"]
        .as_f64()
        .unwrap_or_else(|| panic!("{task}: train_run_wall_s must be a number"));
    if is_media {
        let front = tier["media_front_end_wall_s"].as_f64().unwrap_or_else(|| {
            panic!("{task}: media_front_end_wall_s must be a number on a media leg")
        });
        assert!(
            front.is_finite() && front > 0.0,
            "{task}: the media front end DID run (this leg decodes real PNG/WAV bytes), so its \
             measured wall must be positive and finite, got {front}"
        );
        assert!(
            front < train_wall,
            "{task}: the front-end timer is measured INSIDE TrainingLoop::run, so it must be \
             strictly less than train_run_wall_s ({front} vs {train_wall})"
        );
    } else {
        assert!(
            tier["media_front_end_wall_s"].is_null(),
            "{task}: a text leg never enters the media front end, so this must be null (not \
             0.0, which would claim a path was timed that never ran), got {}",
            tier["media_front_end_wall_s"]
        );
    }
    // Not merely present but DISTINCT: the train split and the held-out
    // fixture are different row sets here, so a digest helper that ignored
    // its argument (or hashed the manifest twice) would collide.
    if is_media {
        assert_ne!(
            tier["train_media_sha256"], tier["heldout_media_sha256"],
            "{task}: the train and held-out corpora are different row sets; equal digests mean \
             the digest ignored its input"
        );
    }
}

/// The profile's POSITIVE-PROOF equation (issue #421 §D4 item 1; unit-467
/// pressure-test folded advisory), checked LIVE on THIS leg's own real CLI
/// output — the same assertion `finetune_run_smoke.rs`'s
/// `fusible_site_census_satisfies_the_positive_proof_equation_on_a_real_run`
/// applies to tiny_bert/text, extended here to the three media towers this
/// file already drives through the real CLI: `fused + eager == <witnessed
/// census field> × steps_measured`, per key.
///
/// `census`/`steps_measured` are read LIVE off this run's own report, never
/// hardcoded — the equation is proven on whatever the committed fixture and
/// this command's flags actually built, not on a number transcribed from a
/// prior run. (A pressure-test run against this branch's tip measured
/// `htsat_clap_tiny` at census `53/21/8` over `steps_measured=2` — totals
/// `106/42/16` — and `tiny_open_clip` at `4/4/0` for the image tower and
/// `4/3/0` for the text tower; this helper's own equation is what a future
/// regression there would trip, not those specific numbers.)
///
/// `gelu_expects_zero`: OpenCLIP's MLP activation is `quick_gelu`
/// (`jammi-encoders/src/activations.rs`, `open_clip_vision.rs`), which has
/// no fused seam and therefore no `admit` key at all, so its
/// `gelu_seam_calls_per_forward` census is legitimately (and checkably) `0`
/// on BOTH OpenCLIP towers — a real, falsifiable claim, not a skip
/// (`profile_421_merge.py`'s own `A_LEG_FUSED_REQUIRED` doc states the same
/// convention for the pod legs). HTSAT's Swin MLP routes through the house
/// `gelu_erf` seam, so its census (and dispatch totals) must be non-zero.
fn assert_positive_proof_equation(stdout: &str, task: &str, gelu_expects_zero: bool) {
    let report: serde_json::Value =
        serde_json::from_str(stdout).unwrap_or_else(|e| panic!("{task}: report must be JSON: {e}"));
    let tier = &report["tiers"]["finetune_run"];
    // The convention the equation is defined under (see
    // `finetune_run_smoke.rs`'s own doc for why `--epochs 1` is
    // load-bearing): both are pinned by `media_command`, asserted here
    // rather than assumed.
    assert_eq!(tier["grad_accum"], serde_json::json!(1), "{task}");
    assert_eq!(tier["epochs"], serde_json::json!(1), "{task}");
    let steps = tier["steps_measured"]
        .as_u64()
        .unwrap_or_else(|| panic!("{task}: steps_measured must be a number"));
    assert!(
        steps > 0,
        "{task}: steps_measured is 0 — nothing was measured"
    );
    let census = tier["fusible_site_census"]
        .as_object()
        .unwrap_or_else(|| panic!("{task}: fusible_site_census must serialize as an object"));

    for (census_field, fused_field, eager_field, is_gelu) in [
        (
            "lora_sites_wrapped",
            "lora_linear_fused_dispatches",
            "lora_linear_eager_dispatches",
            false,
        ),
        (
            "layer_norms",
            "ln_fused_dispatches",
            "ln_eager_dispatches",
            false,
        ),
        (
            "gelu_seam_calls_per_forward",
            "gelu_fused_dispatches",
            "gelu_eager_dispatches",
            true,
        ),
    ] {
        let calls = census[census_field].as_u64().unwrap_or_else(|| {
            panic!("{task}: census.{census_field} must be a non-negative integer")
        });
        if is_gelu && gelu_expects_zero {
            assert_eq!(
                calls, 0,
                "{task}: census.{census_field} must read 0 on this tower (quick_gelu has no \
                 fused seam) — a non-zero value here means the tower's activation seam changed \
                 and this test's `gelu_expects_zero` premise no longer holds"
            );
        } else {
            // Non-vacuity: on every OTHER key (and on gelu for HTSAT), the
            // census must be genuinely non-zero, so the equality below is a
            // real constraint rather than `0 == 0`.
            assert!(
                calls > 0,
                "{task}: census.{census_field} is 0 — the equation below would be vacuous"
            );
        }
        let fused = tier[fused_field]
            .as_u64()
            .unwrap_or_else(|| panic!("{task}: {fused_field} must be a number"));
        let eager = tier[eager_field]
            .as_u64()
            .unwrap_or_else(|| panic!("{task}: {eager_field} must be a number"));
        assert_eq!(
            fused + eager,
            calls * steps,
            "{task}: positive proof failed: {fused_field}={fused} + {eager_field}={eager} != \
             census.{census_field}={calls} x steps_measured={steps}"
        );
    }
}

/// The OpenCLIP VISION tower, end to end over the fixed-shape image corpus.
#[test]
fn image_embedding_leg_runs_end_to_end_over_the_committed_producer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus");
    if !run_producer(
        "ci/scripts/perf/gen_fixed_shape_image_corpus.py",
        &corpus,
        32,
        // The committed OpenCLIP fixture's own `image_size` is 8, so the
        // corpus is generated at exactly the tower's input shape — the
        // fixed-shape premise this producer exists for.
        &["--size", "8"],
    ) {
        media_producer_require_gate();
        return;
    }
    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");
    let model_dir = repo_root().join("cookbook/fixtures/tiny_open_clip");
    let output = media_command(
        &model_dir,
        "image_embedding",
        &corpus,
        &work_dir,
        CLIP_FULL_TARGET_MODULES,
        "triplet",
    )
    .output()
    .expect("run jammi-bench finetune-run --task image_embedding");
    assert!(
        output.status.success(),
        "image_embedding leg failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_well_formed_media_report(&String::from_utf8_lossy(&output.stdout), "image_embedding");
    // OpenCLIP's `quick_gelu` MLP activation has no fused seam at all, so
    // this tower's gelu census (and dispatch totals) must read 0.
    assert_positive_proof_equation(
        &String::from_utf8_lossy(&output.stdout),
        "image_embedding",
        true,
    );
}

/// The HTSAT AUDIO tower, end to end over the fixed-length clip corpus.
///
/// Deliberately the smallest corpus that still clears the trainer's own
/// validation-split floor (8 rows at `--batch 4`, `--validation-fraction
/// 0.25`) and the shortest clip the CLAP front end folds without a
/// degenerate resample — this leg exists to prove the AUDIO chain runs, and
/// the cost measurement it enables is PR B's job, not this test's.
#[test]
fn audio_embedding_leg_runs_end_to_end_over_the_committed_producer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus");
    if !run_producer(
        "ci/scripts/perf/gen_fixed_length_audio_corpus.py",
        &corpus,
        8,
        &["--seconds", "0.1", "--sample-rate", "16000"],
    ) {
        media_producer_require_gate();
        return;
    }
    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");
    let model_dir = repo_root().join("cookbook/fixtures/htsat_clap_tiny");
    let output = media_command(
        &model_dir,
        "audio_embedding",
        &corpus,
        &work_dir,
        CLAP_FULL_TARGET_MODULES,
        "triplet",
    )
    .output()
    .expect("run jammi-bench finetune-run --task audio_embedding");
    assert!(
        output.status.success(),
        "audio_embedding leg failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_well_formed_media_report(&String::from_utf8_lossy(&output.stdout), "audio_embedding");
    // HTSAT's Swin MLP routes through the house `gelu_erf` seam, unlike
    // either OpenCLIP tower, so its census (and dispatch totals) must be
    // genuinely non-zero.
    assert_positive_proof_equation(
        &String::from_utf8_lossy(&output.stdout),
        "audio_embedding",
        false,
    );
}

/// The SAME OpenCLIP checkpoint's TEXT tower — the sharp pairing with the
/// test above: one `--model-dir`, one family, two towers, selected purely by
/// `--task`. Uses the committed text producer, so this leg also proves the
/// text row shape is untouched by the media work.
#[test]
fn text_embedding_leg_selects_the_clip_text_tower_of_the_same_checkpoint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let jsonl = tmp.path().join("text.jsonl");
    let produced = Command::new("python3")
        .current_dir(repo_root())
        .args([
            "ci/scripts/perf/gen_fixed_width_corpus.py",
            "--rows",
            "32",
            "--min-wordpieces",
            "8",
            "--seed",
            "5",
            "--out",
        ])
        .arg(&jsonl)
        .output();
    let produced = match produced {
        Ok(o) => o,
        Err(e) => {
            eprintln!("python3 is not runnable here ({e}); the CLIP-text leg was NOT exercised");
            media_producer_require_gate();
            return;
        }
    };
    assert!(
        produced.status.success(),
        "gen_fixed_width_corpus failed: {}",
        String::from_utf8_lossy(&produced.stderr)
    );
    // Reuse the media command shape by placing the JSONL in its own dir under
    // the name the helper expects.
    let corpus = tmp.path().join("corpus");
    std::fs::create_dir_all(&corpus).expect("mkdir corpus");
    std::fs::copy(&jsonl, corpus.join("triplets.jsonl")).expect("place text jsonl");

    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");
    let model_dir = repo_root().join("cookbook/fixtures/tiny_open_clip");
    let output = media_command(
        &model_dir,
        "text_embedding",
        &corpus,
        &work_dir,
        CLIP_FULL_TARGET_MODULES,
        "triplet",
    )
    .args(["--max-seq-length", "16"])
    .output()
    .expect("run jammi-bench finetune-run --task text_embedding on tiny_open_clip");
    assert!(
        output.status.success(),
        "clip-text leg failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_well_formed_media_report(&String::from_utf8_lossy(&output.stdout), "text_embedding");
    // The SAME checkpoint's TEXT tower is a DIFFERENT encoder than its
    // vision tower above (different layer/site counts), but shares the
    // same `quick_gelu` activation — no fused seam either.
    assert_positive_proof_equation(
        &String::from_utf8_lossy(&output.stdout),
        "text_embedding",
        true,
    );
}

/// Negative control on the whole CLI path, not just the in-process dispatch:
/// pointing `--task image_embedding` at a corpus of TEXT rows must fail with
/// a message naming the missing media field — never silently train on an
/// empty row set.
#[test]
fn a_text_corpus_under_a_media_task_is_refused_by_the_cli() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus");
    std::fs::create_dir_all(&corpus).expect("mkdir corpus");
    std::fs::write(
        corpus.join("triplets.jsonl"),
        "{\"anchor_id\":\"a0\",\"anchor_text\":\"x\",\"positive_id\":\"p0\",\
         \"positive_text\":\"y\",\"negative_id\":\"n0\",\"negative_text\":\"z\"}\n",
    )
    .expect("write text jsonl");
    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");
    let model_dir = repo_root().join("cookbook/fixtures/tiny_open_clip");
    let output = media_command(
        &model_dir,
        "image_embedding",
        &corpus,
        &work_dir,
        "in_proj,c_fc",
        "triplet",
    )
    .output()
    .expect("run jammi-bench finetune-run");
    assert!(
        !output.status.success(),
        "a text corpus under --task image_embedding must fail, not run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("anchor_path"),
        "the refusal must name the media field the row lacks: {stderr}"
    );
}

/// `--objective mnrl` under a MEDIA task is refused (`finetune_run.rs`'s
/// `RowSet::Media` × `Objective::Mnrl` arm) — the trainer's media loader
/// carries the (anchor, positive, negative) triplet shape only, so running
/// the triplet loss under an MNRL label would make the leg unpairable with
/// every real MNRL leg. Driven through the real CLI, and paired with the
/// three tests above (which prove `--objective triplet` IS accepted for all
/// three tasks): without that pairing this assertion would also be
/// satisfied by a build that refused every objective for every media task.
#[test]
fn a_media_task_under_the_mnrl_objective_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let corpus = tmp.path().join("corpus");
    if !run_producer(
        "ci/scripts/perf/gen_fixed_shape_image_corpus.py",
        &corpus,
        8,
        &["--size", "8"],
    ) {
        media_producer_require_gate();
        return;
    }
    let work_dir = tmp.path().join("work");
    std::fs::create_dir_all(&work_dir).expect("mkdir work");
    let model_dir = repo_root().join("cookbook/fixtures/tiny_open_clip");
    let output = media_command(
        &model_dir,
        "image_embedding",
        &corpus,
        &work_dir,
        "in_proj,c_fc",
        "mnrl",
    )
    .output()
    .expect("run jammi-bench finetune-run --task image_embedding --objective mnrl");
    assert!(
        !output.status.success(),
        "--objective mnrl under a media task must be refused, not run:\nstdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mnrl") && stderr.contains("--objective triplet"),
        "the refusal must name the objective it rejected AND the one to use: {stderr}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).trim().is_empty(),
        "a refused invocation must emit no report at all"
    );
}
