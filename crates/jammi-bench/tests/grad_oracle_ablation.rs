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

fn base_command(model_dir: &Path, out: &Path, lora_init: &str) -> Command {
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
        "--seed",
        "11",
        "--lora-init",
        lora_init,
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

/// The load-bearing end-to-end case: a real `--ablate-each-op` run produces
/// a report with `all_fused`, `f32_truth`, `all_off`, and at least one
/// `ablate:<key>` arm, every arm's `vacuous_tensor_count == 0` and
/// `unmatched_disables` empty, and `f32_truth` scores a cosine of
/// (numerically) `1.0` against itself.
#[test]
fn ablate_each_op_end_to_end_on_cpu_fixture() {
    let dir = model_dir();
    assert!(
        dir.join("config.json").exists(),
        "fixture missing: {}",
        dir.display()
    );
    let out = scratch_out("report.json");

    let output = base_command(&dir, &out, "gaussian")
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
    // Every declared ablated_op_key must have a matching arm, and vice
    // versa for the per-op arms.
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

    for arm in arms {
        let name = arm["arm"].as_str().unwrap();
        assert_eq!(
            arm["vacuous_tensor_count"], 0,
            "arm {name:?} has a nonzero vacuous_tensor_count -- Gaussian init should make every \
             gradient live on every arm"
        );
        assert!(
            arm["unmatched_disables"]
                .as_array()
                .expect("unmatched_disables array")
                .is_empty(),
            "arm {name:?} has a non-empty unmatched_disables -- provenance not self-describing"
        );
        assert!(
            arm["matched_tensor_count"].as_u64().unwrap() > 0,
            "arm {name:?} matched zero tensors against f32_truth"
        );
    }

    let f32_truth_arm = arms
        .iter()
        .find(|a| a["arm"] == "f32_truth")
        .expect("f32_truth arm");
    let f32_truth_cosine = f32_truth_arm["overall_cosine_vs_f32_truth"]
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
    let all_fused_cosine = all_fused_arm["overall_cosine_vs_f32_truth"]
        .as_f64()
        .unwrap();
    assert!(
        all_fused_cosine > 0.999,
        "all_fused (f32 backbone) vs f32_truth should agree near-perfectly on this fixture, got \
         {all_fused_cosine}"
    );

    let _ = std::fs::remove_file(&out);
}

/// NEGATIVE CONTROL (family F): `LoraInitMode::ZerosB` leaves `dL/dA`
/// structurally vacuous, so an `--ablate-each-op` run at `--lora-init
/// zeros-b` must REFUSE (nonzero exit, no report written) rather than
/// silently emit a comparison that carries no evidence about half the
/// adapter's own gradients.
#[test]
fn ablate_each_op_refuses_a_structurally_vacuous_fixture() {
    let dir = model_dir();
    let out = scratch_out("report-vacuous.json");

    let output = base_command(&dir, &out, "zeros-b")
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
