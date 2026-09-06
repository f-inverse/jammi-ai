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
/// `--model-dir` and `--target-modules`.
fn media_command(
    model_dir: &Path,
    task: &str,
    corpus: &Path,
    work_dir: &Path,
    target_modules: &str,
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
            "triplet",
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
        "query,value",
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
        "in_proj,c_fc",
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
