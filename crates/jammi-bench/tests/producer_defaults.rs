//! Every producer, run at its defaults over the committed tiny fixtures,
//! files legs the comparator reads on every axis. A producer whose default
//! series is shorter than the ladder's minimum, or whose default leg lacks
//! a field an axis reads, is a defect here, never at the GPU session that
//! would otherwise be the first to find it.

use std::path::Path;
use std::process::Command;

fn bench(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(args)
        .output()
        .expect("run jammi-bench");
    assert!(
        output.status.success(),
        "jammi-bench {} failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The ladder over `legs`, speed axis alone: its verdict must carry no
/// series refusal — the producer's default run is long enough to be read.
/// Its exit status is not the question: a tiny CPU fixture's speed edge may
/// read red or grey, and still be read.
fn speed_axis_reads(workload: &str, legs: &Path, from: &str, to: &str) {
    let out = tempfile::tempdir().expect("verdict dir");
    let ladder = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(["ladder", workload])
        .arg(legs)
        .args([
            "--from",
            from,
            "--to",
            to,
            "--axes",
            "speed",
            "--waive-control",
            "--out",
        ])
        .arg(out.path())
        .output()
        .expect("run the ladder");
    let written = |file: &str| {
        std::fs::read_to_string(out.path().join(file)).unwrap_or_else(|e| {
            panic!(
                "{workload}: the ladder wrote no {file} ({e}):\n{}",
                String::from_utf8_lossy(&ladder.stderr)
            )
        })
    };
    let verdict = written("ladder_verdict.json");
    let table = written("ladder_table.txt");
    assert!(
        !verdict.contains("timed iterations") && !table.contains("timed iterations"),
        "{workload}: a default run's series is too short for the speed axis:\n{table}"
    );
    assert!(
        !table.contains("no leg carries iter_wall_s"),
        "{workload}: a default leg carries no timed series:\n{table}"
    );
}

#[test]
fn encode_step_at_its_defaults_is_read_on_the_speed_axis() {
    let legs = tempfile::tempdir().expect("legs dir");
    let exchange = tempfile::tempdir().expect("exchange dir");
    bench(&[
        "encode-step",
        "--rung",
        "direct",
        "--rung",
        "plan",
        "--rung",
        "plan-partitioned",
        "--rows",
        "16",
        "--legs-dir",
        legs.path().to_str().unwrap(),
        "--exchange-dir",
        exchange.path().to_str().unwrap(),
    ]);
    speed_axis_reads("encode", legs.path(), "direct", "plan-partitioned");
}

#[test]
fn propagate_at_its_defaults_is_read_on_the_speed_axis() {
    let legs = tempfile::tempdir().expect("legs dir");
    bench(&[
        "propagate",
        "--nodes",
        "32",
        "--legs-dir",
        legs.path().to_str().unwrap(),
    ]);
    speed_axis_reads("propagate", legs.path(), "plan", "plan-partitioned");
}

#[test]
fn predictor_train_run_at_its_defaults_is_read_on_the_speed_axis() {
    let legs = tempfile::tempdir().expect("legs dir");
    bench(&[
        "predictor-train-run",
        "--legs-dir",
        legs.path().to_str().unwrap(),
    ]);
    speed_axis_reads(
        "predictor-train-run",
        legs.path(),
        "in-process",
        "in-process",
    );
}

/// A run naming one take of one point is that single point, filed as that
/// take: the invocation a sweep hands each of its points never sweeps again.
#[test]
fn a_run_naming_one_take_of_one_point_files_that_leg_alone() {
    let legs = tempfile::tempdir().expect("legs dir");
    let filed: Vec<String> = serde_json::from_str(&bench(&[
        "propagate",
        "--nodes",
        "32",
        "--rung",
        "plan",
        "--take",
        "3",
        "--iterations",
        "1",
        "--legs-dir",
        legs.path().to_str().unwrap(),
    ]))
    .expect("the producer prints the legs it filed");
    assert_eq!(filed.len(), 1, "one point, one leg: {filed:?}");
    assert!(
        filed[0].ends_with("__r3.json"),
        "filed as take 3: {filed:?}"
    );
}
