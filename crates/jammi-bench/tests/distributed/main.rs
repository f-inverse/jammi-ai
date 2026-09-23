//! The ladder's rungs above the in-process ones, on a fleet this process
//! spawns against the lane's Postgres and S3-class store: every train-run
//! rung — `resident`, `streamed`, `placed`, one-host `shape-d` — publishes
//! the SAME adapter digest on the tiny fixture; every encode rung — `plan`,
//! `placed`, one-host `shape-d` — the SAME vectors digest; a placed
//! propagation the SAME vectors as the plan; and every predictor-train-run
//! rung — `in-process`, `placed`, one-host `shape-d` — the SAME predictions
//! and final weights. Each leg records where its work ran, and a leg above
//! the in-process rungs ran on a process that was not the submitter.
//!
//! Needs the `jammi-server` binary built with `storage-s3` into this
//! target dir (`jammi_test_utils::fleet::jammi_server_binary`) and the
//! lane's backends in the environment (`JAMMI_TEST_PG_URL`,
//! `JAMMI_TEST_S3_ENDPOINT`, `JAMMI_TEST_S3_BUCKET`, `AWS_*`).

use std::path::{Path, PathBuf};
use std::process::Command;

use jammi_test_utils::fleet::jammi_server_binary;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

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

fn run_train_rung(rung: &str, fixtures: &Path, work_dir: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(["finetune-run", "--model-dir"])
        .arg(model_dir())
        .args(["--rung", rung])
        .arg("--server-bin")
        .arg(jammi_server_binary())
        .arg("--train-jsonl")
        .arg(fixtures.join("train.jsonl"))
        .arg("--heldout-ids")
        .arg(fixtures.join("heldout_ids.txt"))
        .arg("--heldout-jsonl")
        .arg(fixtures.join("heldout.jsonl"))
        .args([
            "--epochs",
            "2",
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
            "0.05",
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
        "finetune-run --rung {rung} failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("finetune-run emits a JSON report");
    report["tiers"]["finetune_run"].clone()
}

#[test]
fn every_train_run_rung_publishes_the_same_adapter() {
    let fixtures = tempfile::tempdir().expect("fixtures tempdir");
    write_triplets_jsonl(fixtures.path(), "train.jsonl", 6, 0);
    write_triplets_jsonl(fixtures.path(), "heldout.jsonl", 2, 100);
    write_heldout_ids(fixtures.path(), 2, 100);

    let rungs = ["resident", "streamed", "placed", "shape-d"];
    let mut legs = Vec::new();
    for rung in rungs {
        let work_dir = tempfile::tempdir().expect("work dir");
        legs.push((rung, run_train_rung(rung, fixtures.path(), work_dir.path())));
    }
    let digests: Vec<&str> = legs
        .iter()
        .map(|(_, leg)| leg["outcome_digest"].as_str().expect("outcome digest"))
        .collect();
    for (rung, leg) in &legs {
        eprintln!(
            "{rung}: outcome_digest={} held_out={} ran_on={} stations={{claim_latency_s: {}, \
             placement_s: {}, materialization_s: {}, publish_s: {}}}",
            leg["outcome_digest"],
            leg["held_out_example_mean"],
            leg["ran_on"],
            leg["claim_latency_s"],
            leg["placement_s"],
            leg["materialization_s"],
            leg["publish_s"],
        );
    }
    assert!(
        digests.iter().all(|d| *d == digests[0]),
        "every rung must publish the same adapter: {:?}",
        legs.iter()
            .map(|(rung, leg)| (rung, leg["outcome_digest"].clone()))
            .collect::<Vec<_>>()
    );
    let trajectory = |leg: &serde_json::Value| -> Vec<(u64, f64, String)> {
        leg["trajectory"]
            .as_array()
            .expect("trajectory")
            .iter()
            .map(|p| {
                (
                    p["epoch"].as_u64().unwrap(),
                    p["held_out_mean"].as_f64().unwrap(),
                    p["held_out_batch_partition_sha256"].to_string(),
                )
            })
            .collect()
    };
    for (rung, leg) in &legs {
        assert_eq!(leg["rung"], *rung);
        assert_eq!(
            leg["held_out_example_mean"],
            legs[0].1["held_out_example_mean"]
        );
        assert_eq!(trajectory(leg), trajectory(&legs[0].1));
        assert_eq!(
            leg["initial_adapter_sha256"],
            legs[0].1["initial_adapter_sha256"]
        );
    }
    let role = |leg: &serde_json::Value| leg["ran_on"]["role"].as_str().unwrap().to_string();
    assert_eq!(role(&legs[0].1), "bench");
    assert_eq!(role(&legs[1].1), "worker");
    assert_eq!(
        role(&legs[2].1),
        "executor",
        "a placed leg trained on the executor"
    );
    assert_eq!(
        role(&legs[3].1),
        "compute",
        "a shape-d leg trained on a compute process"
    );
    for (rung, leg) in &legs[2..] {
        let evidence = leg["ran_on"]["evidence"].as_array().expect("evidence");
        assert!(
            evidence.len() >= 2,
            "the {rung} leg carries the lines that prove where it ran: {evidence:?}"
        );
        assert!(leg["placement_s"].is_number(), "{rung} times its placement");
        assert!(leg["claim_latency_s"].is_number());
        assert!(leg["materialization_s"].is_number());
        assert!(leg["publish_s"].is_number());
    }
    assert!(
        legs[2].1["placement_s"].as_f64().unwrap() > 0.0,
        "a placed attempt begins on the executor after its claim"
    );
}

/// One `jammi-bench` subcommand, on the fleet the lane's server binary
/// spawns; its standard output.
fn bench(args: &[&str], legs_dir: &Path) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(args)
        .arg("--legs-dir")
        .arg(legs_dir)
        .arg("--server-bin")
        .arg(jammi_server_binary())
        .output()
        .expect("run jammi-bench");
    assert!(
        output.status.success(),
        "jammi-bench {} failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// The leg filed as `<stem>.json` under `legs_dir`, at its tier.
fn filed_leg(legs_dir: &Path, stem: &str, tier: &str) -> serde_json::Value {
    let path = legs_dir.join(format!("{stem}.json"));
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("a leg is JSON");
    assert!(
        legs_dir.join(format!("{stem}.vectors.f32")).is_file(),
        "the leg's vectors sit beside it"
    );
    report["tiers"][tier].clone()
}

fn run_encode_rung(rung: &str, legs_dir: &Path) -> serde_json::Value {
    bench(
        &[
            "encode-step",
            "--rung",
            rung,
            "--rows",
            "8",
            "--takes",
            "1",
            "--seed",
            "3",
            "--warmup",
            "1",
            "--iters",
            "2",
        ],
        legs_dir,
    );
    filed_leg(legs_dir, &format!("{rung}__rows8__r1"), "encode_step")
}

#[test]
fn every_encode_rung_serves_the_same_vectors() {
    let legs_dir = tempfile::tempdir().expect("legs dir");
    let rungs = ["plan", "placed", "shape-d"];
    let legs: Vec<(&str, serde_json::Value)> = rungs
        .iter()
        .map(|&rung| (rung, run_encode_rung(rung, legs_dir.path())))
        .collect();
    for (rung, leg) in &legs {
        eprintln!(
            "{rung}: outcome_digest={} ran_on={} iter_wall_s={}",
            leg["outcome_digest"], leg["ran_on"], leg["iter_wall_s"]
        );
    }
    let digests: Vec<&str> = legs
        .iter()
        .map(|(_, leg)| leg["outcome_digest"].as_str().expect("outcome digest"))
        .collect();
    assert!(
        digests.iter().all(|d| *d == digests[0]),
        "every encode rung must serve the same vectors: {digests:?}"
    );
    assert!(
        legs[0].1["ran_on"].is_null(),
        "a plan leg never left its process"
    );
    assert_eq!(legs[1].1["ran_on"]["role"], "executor");
    assert_eq!(legs[2].1["ran_on"]["role"], "compute");
    for (rung, leg) in &legs[1..] {
        assert!(
            leg["ran_on"]["evidence"]
                .as_array()
                .is_some_and(|e| e.len() >= 2),
            "the {rung} leg carries the lines that prove where it ran: {}",
            leg["ran_on"]
        );
    }
    assert_eq!(legs[0].1["vector_dim"], legs[2].1["vector_dim"]);
}

/// The placed propagation's sink is written by the executor, and the
/// vectors it writes are the plan's bytes.
#[test]
fn a_placed_propagation_matches_the_plan() {
    let legs_dir = tempfile::tempdir().expect("legs dir");
    let files: Vec<String> = serde_json::from_slice(&bench(
        &[
            "propagate",
            "--nodes",
            "32",
            "--rung",
            "plan,placed",
            "--warmup",
            "0",
            "--iterations",
            "2",
        ],
        legs_dir.path(),
    ))
    .expect("propagate prints its leg files");
    let stem = |rung: &str| {
        files
            .iter()
            .find_map(|f| f.strip_prefix(&format!("{rung}__"))?.strip_suffix(".json"))
            .map(|unit_take| format!("{rung}__{unit_take}"))
            .unwrap_or_else(|| panic!("no {rung} leg among {files:?}"))
    };
    let plan = filed_leg(legs_dir.path(), &stem("plan"), "propagate");
    let placed = filed_leg(legs_dir.path(), &stem("placed"), "propagate");
    eprintln!(
        "plan: outcome_digest={} iter_wall_s={}\nplaced: outcome_digest={} iter_wall_s={} \
         ran_on={}",
        plan["outcome_digest"],
        plan["iter_wall_s"],
        placed["outcome_digest"],
        placed["iter_wall_s"],
        placed["ran_on"]
    );
    assert_eq!(
        placed["outcome_digest"], plan["outcome_digest"],
        "the placed propagation writes the plan's bytes"
    );
    assert_eq!(placed["unit"], plan["unit"]);
    assert_eq!(placed["rung"], "placed");
    assert!(
        plan["ran_on"].is_null(),
        "a plan leg never left its process"
    );
    assert_eq!(
        placed["ran_on"]["role"], "executor",
        "the placed sink was written by the executor: {}",
        placed["ran_on"]
    );
    assert!(
        placed["ran_on"]["evidence"]
            .as_array()
            .is_some_and(|e| e.len() >= 3),
        "the placement line, the binding and the sink write: {}",
        placed["ran_on"]
    );
    assert_eq!(placed["iter_wall_s"].as_array().map(Vec::len), Some(2));
}

/// The predictor a placed job and a shape-d job publish is the one the
/// in-process fit trains: the same predictions, the same final weights, the
/// same learning curve — trained on an executor and on a compute process,
/// never on the submitter.
#[test]
fn every_predictor_train_run_rung_publishes_the_same_predictor() {
    let legs_dir = tempfile::tempdir().expect("legs dir");
    bench(
        &[
            "predictor-train-run",
            "--rung",
            "in-process,placed,shape-d",
            "--seeds",
            "7",
            "--epochs",
            "2",
            "--warmup-steps",
            "0",
        ],
        legs_dir.path(),
    );
    let rungs = ["in-process", "placed", "shape-d"];
    let legs: Vec<(&str, serde_json::Value)> = rungs
        .iter()
        .map(|&rung| {
            (
                rung,
                filed_leg(
                    legs_dir.path(),
                    &format!("{rung}__seed7__r1"),
                    "predictor_train_run",
                ),
            )
        })
        .collect();
    let curve = |leg: &serde_json::Value| {
        (
            leg["held_out_at_init"].clone(),
            leg["trajectory"]
                .as_array()
                .expect("trajectory")
                .iter()
                .map(|p| (p["epoch"].clone(), p["held_out_mean"].clone()))
                .collect::<Vec<_>>(),
            leg["train_probe_series"].clone(),
        )
    };
    for (rung, leg) in &legs {
        eprintln!(
            "{rung}: outcome_digest={} final_weights={} held_out={} ran_on={} stations={{\
             claim_latency_s: {}, placement_s: {}, publish_s: {}}}",
            leg["outcome_digest"],
            leg["final_weights"]["sha256"],
            leg["held_out_example_mean"],
            leg["ran_on"],
            leg["claim_latency_s"],
            leg["placement_s"],
            leg["publish_s"],
        );
        assert_eq!(leg["rung"], *rung);
        assert_eq!(leg["outcome_digest"], legs[0].1["outcome_digest"], "{rung}");
        assert_eq!(
            leg["final_weights"]["sha256"], legs[0].1["final_weights"]["sha256"],
            "{rung}"
        );
        assert_eq!(
            leg["initial_weights_sha256"], legs[0].1["initial_weights_sha256"],
            "{rung}"
        );
        assert_eq!(curve(leg), curve(&legs[0].1), "{rung}");
        assert_eq!(leg["iters_measured"], legs[0].1["iters_measured"], "{rung}");
    }
    assert!(
        legs[0].1["ran_on"].is_null(),
        "an in-process leg never left its process"
    );
    assert_eq!(
        legs[1].1["ran_on"]["role"], "executor",
        "{}",
        legs[1].1["ran_on"]
    );
    assert_eq!(
        legs[2].1["ran_on"]["role"], "compute",
        "{}",
        legs[2].1["ran_on"]
    );
    for (rung, leg) in &legs[1..] {
        assert!(
            leg["ran_on"]["evidence"]
                .as_array()
                .is_some_and(|e| !e.is_empty()),
            "the {rung} leg carries what proves where it ran: {}",
            leg["ran_on"]
        );
        assert!(leg["claim_latency_s"].is_number(), "{rung} times its claim");
        assert!(leg["placement_s"].is_number(), "{rung} times its placement");
        assert!(leg["publish_s"].is_number(), "{rung} times its publish");
    }
    assert!(
        legs[1].1["placement_s"].as_f64().unwrap() > 0.0,
        "a placed attempt begins on the executor after its claim"
    );
}
