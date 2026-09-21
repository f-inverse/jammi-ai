//! `jammi-bench kernel-arm` through the built binary: the census taken to
//! its fixpoint over absorption, on the two tiny text checkpoints. A fresh
//! process per pass is what lets the block kernel be turned off for the
//! second pass, so the keys it absorbs are reached.

use std::path::PathBuf;
use std::process::Command;

/// Every key a family switches, as the arm derivation knows them.
struct KernelFamilyKeys;

impl KernelFamilyKeys {
    const ALL: [&'static str; 14] = [
        "layer_norm_fused",
        "attention_block_flash",
        "mem_efficient_attention",
        "attention_block_fused",
        "rope_fused",
        "softmax_last_dim_fused",
        "geglu_fused",
        "gelu_erf_fused",
        "lora_linear_fused",
        "cast_scale_bf16_f32",
        "cast_scale_f16_f32",
        "cast_add_bf16",
        "cast_add_f16",
        "adamw_step_fused",
    ];
}

fn fixture(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn derive(model_dir: &PathBuf, target_modules: &str, args: &[&str]) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(["kernel-arm", "--model-dir"])
        .arg(model_dir)
        .args(["--target-modules", target_modules, "--json"])
        .args(args)
        .env_remove("JAMMI_KERNELS_DISABLE")
        .env_remove("JAMMI_KERNELS_STRICT")
        .output()
        .expect("spawn jammi-bench kernel-arm");
    assert!(
        output.status.success(),
        "kernel-arm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("kernel-arm prints JSON")
}

fn strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn the_all_off_arm_reaches_what_the_block_kernel_absorbs_on_the_tower_that_has_it() {
    let modernbert = derive(
        &fixture("../jammi-encoders/tests/fixtures/tiny_modernbert_head64"),
        "Wqkv,Wo",
        &["--all"],
    );
    let bert = derive(
        &fixture("../../cookbook/fixtures/tiny_bert_head64"),
        "query,value",
        &["--all"],
    );
    let (m, b) = (strings(&modernbert["disable"]), strings(&bert["disable"]));
    eprintln!("modernbert all-off: {m:?}\nbert all-off: {b:?}");
    // The towers differ by exactly the seams each holds: RoPE and GeGLU on
    // ModernBERT, GELU-erf on BERT.
    for key in ["rope_fused", "geglu_fused"] {
        assert!(m.iter().any(|k| k == key), "{key} missing from {m:?}");
        assert!(!b.iter().any(|k| k == key), "{key} present in {b:?}");
    }
    // The ModernBERT set is the one a committed A100 sweep's every-family-off
    // legs recorded as fired.
    assert_eq!(
        m,
        [
            "adamw_step_fused",
            "attention_block_flash",
            "attention_block_fused",
            "geglu_fused",
            "layer_norm_fused",
            "lora_linear_fused",
            "mem_efficient_attention",
            "rope_fused",
            "softmax_last_dim_fused"
        ]
    );
    assert!(b.iter().any(|k| k == "gelu_erf_fused"));
    assert!(!m.iter().any(|k| k == "gelu_erf_fused"));
    for key in m.iter().chain(&b) {
        assert!(
            KernelFamilyKeys::ALL.contains(&key.as_str()),
            "{key} is consulted but belongs to no family"
        );
    }
    for shared in [
        "adamw_step_fused",
        "attention_block_flash",
        "attention_block_fused",
        "layer_norm_fused",
        "lora_linear_fused",
        "mem_efficient_attention",
        "softmax_last_dim_fused",
    ] {
        assert!(
            m.iter().any(|k| k == shared) && b.iter().any(|k| k == shared),
            "{shared}"
        );
    }
    let sorted = |v: &[String]| {
        let mut s = v.to_vec();
        s.sort();
        s == v
    };
    assert!(sorted(&m) && sorted(&b));
}

#[test]
fn the_how_well_reference_arm_is_the_two_families_on_either_tower() {
    for (dir, targets) in [
        (
            fixture("../jammi-encoders/tests/fixtures/tiny_modernbert_head64"),
            "Wqkv,Wo",
        ),
        (
            fixture("../../cookbook/fixtures/tiny_bert_head64"),
            "query,value",
        ),
    ] {
        let derived = derive(&dir, targets, &["--off", "flash-attention,adam-w"]);
        assert_eq!(
            strings(&derived["disable"]),
            ["adamw_step_fused", "attention_block_flash"]
        );
    }
}

/// A family absorbed by one the arm leaves on is refused by name rather
/// than derived to a key that would never fire on the device.
#[test]
fn an_arm_that_turns_off_an_absorbed_family_alone_is_refused() {
    let output = Command::new(env!("CARGO_BIN_EXE_jammi-bench"))
        .args(["kernel-arm", "--model-dir"])
        .arg(fixture(
            "../jammi-encoders/tests/fixtures/tiny_modernbert_head64",
        ))
        .args(["--target-modules", "Wqkv,Wo", "--off", "rope"])
        .env_remove("JAMMI_KERNELS_DISABLE")
        .output()
        .expect("spawn jammi-bench kernel-arm");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("absorbed by AttentionBlock"), "{stderr}");
}
