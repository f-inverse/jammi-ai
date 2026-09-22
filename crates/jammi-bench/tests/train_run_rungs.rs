//! The train-run ladder's first exact edge, proven on CPU over the tiny
//! synthetic fixture: the `resident` rung (the trainer over in-memory
//! rows) and the `streamed` rung (the engine's job path — a registered
//! source, a submitted job, a materialised training set streamed back, an
//! adapter published through the artifact store) publish a BYTE-IDENTICAL
//! adapter, at dropout 0 and at dropout 0.05, and score the same held-out
//! trajectory off it. Two rungs of one unit differ only in provenance.

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

/// Rows written OUT of the training set's committed order, so a rung that
/// fed the file's order would not match one that fed the engine's.
fn write_triplets_jsonl(dir: &Path, name: &str, n: usize, offset: usize) -> PathBuf {
    let path = dir.join(name);
    let mut body = String::new();
    for i in (0..n).rev() {
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

fn write_heldout_ids(dir: &Path, n: usize, offset: usize) -> PathBuf {
    let path = dir.join("heldout_ids.txt");
    let mut body = String::new();
    for i in (0..n).rev() {
        let k = offset + i;
        body.push_str(&format!("a{k}\tp{k}\tn{k}\n"));
    }
    std::fs::write(&path, body).expect("write heldout ids");
    path
}

struct Fixture {
    train_jsonl: PathBuf,
    heldout_jsonl: PathBuf,
    heldout_ids: PathBuf,
}

fn fixture(dir: &Path) -> Fixture {
    Fixture {
        train_jsonl: write_triplets_jsonl(dir, "train.jsonl", 6, 0),
        heldout_jsonl: write_triplets_jsonl(dir, "heldout.jsonl", 2, 100),
        heldout_ids: write_heldout_ids(dir, 2, 100),
    }
}

fn run_rung(
    rung: &str,
    fixture: &Fixture,
    work_dir: &Path,
    lora_dropout: &str,
) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(["finetune-run", "--model-dir"])
        .arg(model_dir())
        .args(["--arm", "fused", "--rung", rung])
        .arg("--train-jsonl")
        .arg(&fixture.train_jsonl)
        .arg("--heldout-ids")
        .arg(&fixture.heldout_ids)
        .arg("--heldout-jsonl")
        .arg(&fixture.heldout_jsonl)
        .args([
            "--epochs",
            "3",
            "--seed",
            "7",
            "--eval-cadence",
            "1",
            "--batch",
            "2",
            "--lr",
            "0.01",
            "--schedule",
            "cosine_decay",
            "--warmup-steps",
            "1",
            "--weight-decay",
            "0.01",
            "--grad-accum",
            "1",
            "--validation-fraction",
            "0.34",
            "--early-stopping-patience",
            "10000",
            "--early-stopping-metric",
            "val_loss",
            "--max-grad-norm",
            "1.0",
            "--objective",
            "mnrl",
            "--margin",
            "0.3",
            "--temperature",
            "20.0",
            "--lora-rank",
            "2",
            "--lora-alpha",
            "4",
            "--lora-dropout",
            lora_dropout,
            "--target-modules",
            "query,value",
            "--backbone-dtype",
            "f32",
            "--max-seq-length",
            "16",
        ])
        .arg("--work-dir")
        .arg(work_dir)
        .output()
        .expect("run jammi-bench finetune-run");
    assert!(
        output.status.success(),
        "finetune-run --rung {rung} --lora-dropout {lora_dropout} failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("finetune-run emits a JSON report");
    report["tiers"]["finetune_run"].clone()
}

fn assert_exact_edge(lora_dropout: &str) {
    let fixtures_dir = tempfile::tempdir().expect("fixtures tempdir");
    let fixture = fixture(fixtures_dir.path());
    let resident_dir = tempfile::tempdir().expect("resident work dir");
    let streamed_dir = tempfile::tempdir().expect("streamed work dir");
    let resident = run_rung("resident", &fixture, resident_dir.path(), lora_dropout);
    let streamed = run_rung("streamed", &fixture, streamed_dir.path(), lora_dropout);

    assert_eq!(resident["rung"], "resident");
    assert_eq!(streamed["rung"], "streamed");
    assert_eq!(streamed["ran_on"]["role"], "worker");
    assert_eq!(resident["ran_on"]["role"], "bench");

    let digest = |leg: &serde_json::Value| {
        leg["outcome_digest"]
            .as_str()
            .expect("a leg carries its outcome digest")
            .to_string()
    };
    assert_eq!(digest(&resident).len(), 64, "sha256 hex");
    assert_eq!(
        digest(&resident),
        digest(&streamed),
        "at lora_dropout {lora_dropout} the resident and streamed rungs must publish a \
         byte-identical adapter (resident: {}, streamed: {})",
        serde_json::to_string_pretty(&resident).unwrap(),
        serde_json::to_string_pretty(&streamed).unwrap(),
    );
    assert_eq!(
        resident["initial_adapter_sha256"], streamed["initial_adapter_sha256"],
        "the untrained adapter is a function of the seed on every rung"
    );
    assert_eq!(
        resident["train_token_ids_sha256"],
        streamed["train_token_ids_sha256"]
    );
    assert_eq!(
        resident["held_out_example_mean"], streamed["held_out_example_mean"],
        "the same published adapter scores the same held-out mean"
    );
    let trajectory = |leg: &serde_json::Value| -> Vec<(u64, f64)> {
        leg["trajectory"]
            .as_array()
            .expect("trajectory")
            .iter()
            .map(|p| {
                (
                    p["epoch"].as_u64().unwrap(),
                    p["held_out_mean"].as_f64().unwrap(),
                )
            })
            .collect()
    };
    assert_eq!(trajectory(&resident), trajectory(&streamed));
    assert_eq!(trajectory(&resident).len(), 3);
    assert_eq!(
        resident["train_probe_series"],
        streamed["train_probe_series"]
    );
    assert_eq!(resident["steps_measured"], streamed["steps_measured"]);
    assert_eq!(
        resident["epoch_walls"].as_array().map(Vec::len),
        Some(3),
        "one wall per epoch"
    );
    assert_eq!(
        resident["iter_wall_s"].as_array().map(Vec::len),
        resident["steps_measured"].as_u64().map(|n| n as usize),
        "one iteration per optimizer step"
    );
    assert_eq!(resident["work"], 6.0);
    // The job path's stations exist on the streamed rung alone.
    assert!(streamed["materialization_s"].is_number());
    assert!(streamed["claim_latency_s"].is_number());
    assert!(streamed["publish_s"].is_number());
    assert!(resident.get("materialization_s").is_none());
    // The kernel dispatch counters come from the run's own window on both.
    assert_eq!(
        resident["lora_linear_fused_dispatches"].as_u64().unwrap()
            + resident["lora_linear_eager_dispatches"].as_u64().unwrap(),
        streamed["lora_linear_fused_dispatches"].as_u64().unwrap()
            + streamed["lora_linear_eager_dispatches"].as_u64().unwrap(),
        "the two rungs run the same forwards"
    );
}

#[test]
fn resident_and_streamed_publish_the_same_adapter_at_dropout_zero() {
    assert_exact_edge("0.0");
}

#[test]
fn resident_and_streamed_publish_the_same_adapter_at_dropout_five_percent() {
    assert_exact_edge("0.05");
}
