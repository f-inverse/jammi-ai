//! End-to-end test of `jammi-bench grad-oracle --ablate-each-op` through the
//! REAL compiled binary (never `grad_oracle_ablation::run` called
//! in-process) — this feature's whole mechanism is spawning `current_exe()`
//! children with `JAMMI_KERNELS_DISABLE`/`JAMMI_KERNELS_STRICT` set per arm,
//! which only means anything driven through a real child-process boundary
//! (see `grad_oracle_ablation.rs`'s module doc, and
//! `finetune_step_kernel_disable.rs`'s own doc for why an in-process
//! `std::env::set_var` test cannot exercise this at all: those two env vars
//! memoize into process-wide `OnceLock`s).
//!
//! Fixture: the SAME `head_dim == 64` checkpoint
//! `finetune_step_kernel_disable.rs` uses — already confirmed (that file's
//! own module doc) to reach `layer_norm_fused` on CPU/F32 in training mode,
//! so this test's `all_fused` arm has at least one real op key to ablate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../jammi-encoders/tests/fixtures/tiny_modernbert_head64")
}

fn base_command(model_dir: &Path, out: &Path, lora_init: &str, seeds: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args([
        "grad-oracle",
        "--model-dir",
        &model_dir.to_string_lossy(),
        "--batch",
        "3",
        "--seq",
        "6",
        "--lora-rank",
        "2",
        "--lora-alpha",
        "4",
        "--target-modules",
        "Wqkv,Wo",
        "--backbone-dtype",
        "f32",
        "--lora-init",
        lora_init,
        "--seeds",
        seeds,
        "--ablate-each-op",
        "--out",
    ])
    .arg(out);
    cmd.env_remove("JAMMI_KERNELS_DISABLE");
    cmd.env_remove("JAMMI_KERNELS_STRICT");
    cmd
}

/// A process-unique scratch path (mirrors `grad_oracle.rs`'s own `tempdir`
/// helper's rationale: `cargo test` runs this crate's integration test
/// binaries in parallel, and two concurrent runs of THIS test file's own
/// `#[test]` fns could otherwise collide on a shared literal filename).
fn scratch_out(name: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "jammi-bench-grad-oracle-ablation-test-{}-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        n
    ))
}

/// The load-bearing end-to-end case (single seed, for CPU test speed): a
/// real `--ablate-each-op` run produces a report with `all_fused`,
/// `f32_truth`, `all_off`, and at least one `ablate:<key>` arm, every arm's
/// `per_seed` samples carrying `vacuous_tensor_count == 0` and
/// `unmatched_disables` empty, and `f32_truth` scores a `full_tensor_cosine`
/// of (numerically) `1.0` against itself.
#[test]
fn ablate_each_op_end_to_end_on_cpu_fixture() {
    let dir = model_dir();
    assert!(
        dir.join("config.json").exists(),
        "fixture missing: {}",
        dir.display()
    );
    let out = scratch_out("report.json");

    let output = base_command(&dir, &out, "gaussian", "11")
        .output()
        .expect("spawn jammi-bench grad-oracle --ablate-each-op");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "ablate-each-op run failed: stdout={stdout} stderr={stderr}"
    );

    let text = std::fs::read_to_string(&out).expect("read ablation report");
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse ablation report");

    assert_eq!(report["tool"], "jammi_grad_oracle_ablation");
    assert_eq!(report["seeds"], serde_json::json!([11]));
    let arms = report["arms"].as_array().expect("arms array");
    let arm_names: Vec<&str> = arms
        .iter()
        .map(|a| a["arm"].as_str().expect("arm name"))
        .collect();
    assert!(arm_names.contains(&"all_fused"), "{arm_names:?}");
    assert!(arm_names.contains(&"f32_truth"), "{arm_names:?}");
    assert!(arm_names.contains(&"all_off"), "{arm_names:?}");
    assert!(
        arm_names.iter().any(|n| n.starts_with("ablate:")),
        "expected at least one per-op ablation arm on this fixture (layer_norm_fused is known \
         to reach admit on it — see finetune_step_kernel_disable.rs's own doc): {arm_names:?}"
    );

    let ablated_op_keys = report["ablated_op_keys"]
        .as_array()
        .expect("ablated_op_keys array");
    assert!(
        !ablated_op_keys.is_empty(),
        "ablated_op_keys must be non-empty on this fixture"
    );
    let ablate_arm_keys: Vec<&str> = arm_names
        .iter()
        .filter_map(|n| n.strip_prefix("ablate:"))
        .collect();
    let declared_keys: Vec<&str> = ablated_op_keys
        .iter()
        .map(|k| k.as_str().expect("op key string"))
        .collect();
    assert_eq!(
        {
            let mut a = ablate_arm_keys.clone();
            a.sort();
            a
        },
        {
            let mut b = declared_keys.clone();
            b.sort();
            b
        }
    );

    // Every arm's classification is one of the documented labels, and the
    // three structural arms carry the expected fixed label.
    for arm in arms {
        let name = arm["arm"].as_str().unwrap();
        let classification = arm["classification"].as_str().unwrap();
        match name {
            "all_fused" => assert_eq!(classification, "reference"),
            "f32_truth" => assert_eq!(classification, "truth"),
            "all_off" => assert_eq!(classification, "control"),
            _ => assert!(
                classification == "neutral" || classification == "divergent",
                "arm {name:?}: unexpected classification {classification:?}"
            ),
        }
        let per_seed = arm["per_seed"].as_array().expect("per_seed array");
        assert_eq!(
            per_seed.len(),
            1,
            "single-seed run must carry exactly 1 per_seed sample"
        );
        for sample in per_seed {
            assert_eq!(
                sample["vacuous_tensor_count"], 0,
                "arm {name:?} has a nonzero vacuous_tensor_count -- Gaussian init should make \
                 every gradient live on every arm"
            );
            assert!(
                sample["unmatched_disables"]
                    .as_array()
                    .expect("unmatched_disables array")
                    .is_empty(),
                "arm {name:?} has a non-empty unmatched_disables -- provenance not self-describing"
            );
            assert!(
                sample["matched_tensor_count"].as_u64().unwrap() > 0,
                "arm {name:?} matched zero tensors against f32_truth"
            );
            assert!(
                !sample["per_tensor"]
                    .as_array()
                    .expect("per_tensor array")
                    .is_empty(),
                "arm {name:?}: per_tensor rows must be committed (round-7 audit finding)"
            );
        }
        // Single-seed run: median == min == max (a degenerate SeedStat).
        let full = &arm["full_tensor_cosine"];
        assert_eq!(full["median"], full["min"]);
        assert_eq!(full["median"], full["max"]);
    }

    let f32_truth_arm = arms
        .iter()
        .find(|a| a["arm"] == "f32_truth")
        .expect("f32_truth arm");
    let f32_truth_cosine = f32_truth_arm["full_tensor_cosine"]["median"]
        .as_f64()
        .unwrap();
    assert!(
        (f32_truth_cosine - 1.0).abs() < 1e-9,
        "f32_truth compared against itself must score cosine ~1.0, got {f32_truth_cosine}"
    );

    // This fixture's CLI ran with --backbone-dtype f32 for EVERY arm (the
    // f32_truth arm is always f32 regardless; here the reference/ablation
    // arms are ALSO f32, since no bf16 rounding separates them from truth
    // at all -- the only source of divergence is kernel COMPOSITION, not
    // dtype). Expect near-perfect agreement, not merely "positive".
    let all_fused_arm = arms.iter().find(|a| a["arm"] == "all_fused").unwrap();
    let all_fused_cosine = all_fused_arm["full_tensor_cosine"]["median"]
        .as_f64()
        .unwrap();
    assert!(
        all_fused_cosine > 0.999,
        "all_fused (f32 backbone) vs f32_truth should agree near-perfectly on this fixture, got \
         {all_fused_cosine}"
    );

    // derived_per_op_budget must be a finite, non-negative number (a
    // single-seed run has a degenerate spread of exactly 0.0 -- max==min).
    let budget = report["derived_per_op_budget"].as_f64().unwrap();
    assert!(budget.is_finite() && budget >= 0.0, "budget={budget}");
    assert_eq!(budget, 0.0, "single-seed spread must be exactly 0.0");

    let _ = std::fs::remove_file(&out);
}

/// Multi-seed aggregation: three DIFFERENT seeds must each independently
/// draw fresh weights/data (the `all_fused` arm's `per_seed` samples must
/// not be bit-identical to each other), and the reported `full_tensor_cosine`
/// `{median, min, max}` must be internally consistent with the raw
/// `per_seed` values (never trust the aggregate without recomputing it
/// here independently).
#[test]
fn ablate_each_op_multi_seed_aggregation_is_internally_consistent() {
    let dir = model_dir();
    let out = scratch_out("report-multiseed.json");

    let output = base_command(&dir, &out, "gaussian", "101,102,103")
        .output()
        .expect("spawn jammi-bench grad-oracle --ablate-each-op --seeds 101,102,103");
    assert!(
        output.status.success(),
        "multi-seed run failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = std::fs::read_to_string(&out).expect("read ablation report");
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse ablation report");
    assert_eq!(report["seeds"], serde_json::json!([101, 102, 103]));

    let arms = report["arms"].as_array().unwrap();
    let all_fused = arms.iter().find(|a| a["arm"] == "all_fused").unwrap();
    let per_seed = all_fused["per_seed"].as_array().unwrap();
    assert_eq!(per_seed.len(), 3);
    let seeds_seen: Vec<i64> = per_seed
        .iter()
        .map(|s| s["seed"].as_i64().unwrap())
        .collect();
    assert_eq!(seeds_seen, vec![101, 102, 103]);

    let cosines: Vec<f64> = per_seed
        .iter()
        .map(|s| s["full_tensor_cosine"].as_f64().unwrap())
        .collect();
    let mut sorted = cosines.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected_min = sorted[0];
    let expected_max = sorted[2];
    let expected_median = sorted[1];

    let agg = &all_fused["full_tensor_cosine"];
    assert!((agg["min"].as_f64().unwrap() - expected_min).abs() < 1e-12);
    assert!((agg["max"].as_f64().unwrap() - expected_max).abs() < 1e-12);
    assert!((agg["median"].as_f64().unwrap() - expected_median).abs() < 1e-12);

    // Different seeds (different weight draws) generally produce
    // DIFFERENT losses across the three per_seed samples -- a fixture
    // that ignored --seeds and reused the same weights/data every time
    // would show all three losses identical.
    let losses: Vec<f64> = per_seed
        .iter()
        .map(|s| s["loss"].as_f64().unwrap())
        .collect();
    assert!(
        losses[0] != losses[1] || losses[1] != losses[2],
        "all three seeds produced the IDENTICAL loss -- looks like --seeds is not actually \
         varying the weights/data: {losses:?}"
    );
}

/// NEGATIVE CONTROL (family F): `zeros-b` leaves `dL/dA` structurally
/// vacuous, so an `--ablate-each-op` run at `--lora-init zeros-b` must
/// REFUSE (nonzero exit, no report written) rather than silently emit a
/// comparison that carries no evidence about half the adapter's own
/// gradients.
#[test]
fn ablate_each_op_refuses_a_structurally_vacuous_fixture() {
    let dir = model_dir();
    let out = scratch_out("report-vacuous.json");

    let output = base_command(&dir, &out, "zeros-b", "11")
        .output()
        .expect("spawn jammi-bench grad-oracle --ablate-each-op --lora-init zeros-b");

    assert!(
        !output.status.success(),
        "a ZerosB fixture must refuse the ablation comparison, not succeed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("vacuous_tensor_count"),
        "the refusal must name vacuous_tensor_count as the reason -- stderr={stderr}"
    );
    assert!(
        !out.exists(),
        "a REFUSED run must never write the --out report file"
    );
}

/// The `peft-step1` operating point (round-7 audit fix, PR #383): default
/// under `--ablate-each-op` when `--lora-init` is omitted, and must ALSO
/// produce a non-vacuous, successful run — proving the one real `AdamW`
/// step actually moved `B` away from zero on this fixture.
#[test]
fn ablate_each_op_peft_step1_is_the_default_and_succeeds() {
    let dir = model_dir();
    let out = scratch_out("report-peftstep1.json");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_jammi-bench"));
    cmd.args([
        "grad-oracle",
        "--model-dir",
        &dir.to_string_lossy(),
        "--batch",
        "3",
        "--seq",
        "6",
        "--lora-rank",
        "2",
        "--lora-alpha",
        "4",
        "--target-modules",
        "Wqkv,Wo",
        "--backbone-dtype",
        "f32",
        "--seeds",
        "11",
        "--ablate-each-op",
        // NOTE: no --lora-init at all -- must resolve to peft-step1.
        "--out",
    ])
    .arg(&out);
    cmd.env_remove("JAMMI_KERNELS_DISABLE");
    cmd.env_remove("JAMMI_KERNELS_STRICT");

    let output = cmd
        .output()
        .expect("spawn jammi-bench grad-oracle --ablate-each-op (no --lora-init)");
    assert!(
        output.status.success(),
        "peft-step1 default run failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&out).expect("read ablation report");
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse ablation report");
    assert_eq!(report["lora_init"], "peft-step1");

    let _ = std::fs::remove_file(&out);
}
