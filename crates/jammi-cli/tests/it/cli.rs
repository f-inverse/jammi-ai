use assert_cmd::Command;
use predicates::prelude::*;

// ─── CLI: sources list, query, explain ───────────────────────────────────────
//
// Each subcommand is invoked via assert_cmd against the compiled binary.
// Uses a temp dir as artifact_dir to avoid polluting the workspace.

#[test]
fn sources_list_output_contains_header() {
    let dir = tempfile::tempdir().unwrap();
    // Write a minimal config pointing at the temp dir
    let config_path = dir.path().join("test.toml");
    std::fs::write(
        &config_path,
        format!("artifact_dir = {:?}\n[gpu]\ndevice = -1\n", dir.path()),
    )
    .unwrap();

    Command::cargo_bin("jammi-cli")
        .unwrap()
        .arg("sources")
        .arg("list")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicates::str::contains("No sources").or(predicates::str::contains("Name")));
}

#[test]
fn query_select_1_returns_result() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("test.toml");
    std::fs::write(
        &config_path,
        format!("artifact_dir = {:?}\n[gpu]\ndevice = -1\n", dir.path()),
    )
    .unwrap();

    Command::cargo_bin("jammi-cli")
        .unwrap()
        .arg("query")
        .arg("SELECT 1 AS value")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .assert()
        .success()
        .stdout(predicates::str::contains("1"));
}

#[test]
fn explain_shows_plan_operators() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("test.toml");
    std::fs::write(
        &config_path,
        format!("artifact_dir = {:?}\n[gpu]\ndevice = -1\n", dir.path()),
    )
    .unwrap();

    Command::cargo_bin("jammi-cli")
        .unwrap()
        .arg("explain")
        .arg("SELECT 1 + 1 AS value")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .assert()
        .success()
        .stdout(
            predicates::str::contains("projection")
                .or(predicates::str::contains("Projection"))
                .or(predicates::str::contains("expr")),
        );
}
