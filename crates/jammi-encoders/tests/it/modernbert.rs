//! ModernBERT encoder integration tests against the `tiny_modernbert_classifier`
//! fixture (hidden_size=32, layers=1, heads=2, intermediate=64, max_pos=128).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarMap;
use jammi_encoders::modernbert::ModernBertConfig;
use jammi_encoders::{ModernBert, Pooling};
use jammi_lora::{LoraBuildConfig, LoraInitMode};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../cookbook/fixtures/tiny_modernbert_classifier")
}

fn load_config() -> ModernBertConfig {
    let config_path = fixture_dir().join("config.json");
    let raw =
        std::fs::read_to_string(&config_path).expect("read tiny_modernbert_classifier config");
    serde_json::from_str(&raw).expect("parse ModernBertConfig")
}

fn weights_path() -> PathBuf {
    fixture_dir().join("model.safetensors")
}

/// Spec section 2.8 test 7: build with target_modules covering every LoRA
/// injection site and assert the trainable-parameter count is exactly what
/// the architecture predicts.
///
/// With `target_modules = ["Wqkv", "Wo"]` and `should_apply_lora`'s
/// suffix-or-equals matching:
/// - `attn.Wqkv` → matches `"Wqkv"` (exact for the target name passed to the
///   match function).
/// - `attn.Wo`  → matches `"Wo"`.
/// - `mlp.Wo`   → matches `"Wo"` (the MLP output site uses the namespaced
///   target name `"mlp.Wo"` whose `ends_with("Wo")` is true).
/// - `mlp.Wi`   → no match.
///
/// That is 3 sites per layer × 2 tensors (A and B) × `num_hidden_layers`.
#[test]
fn modernbert_loads_with_target_modules() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let targets: Vec<String> = vec!["Wqkv".into(), "Wo".into()];
    let no_layers: Option<Vec<usize>> = None;
    let empty_pattern: HashMap<String, usize> = HashMap::new();
    let lora = LoraBuildConfig {
        target_modules: &targets,
        layers_to_transform: &no_layers,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &empty_pattern,
        init_mode: LoraInitMode::ZerosB,
    };

    let model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .lora(lora)
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build LoRA-targeted ModernBert on tiny_modernbert_classifier");

    assert_eq!(model.hidden_size(), config.hidden_size);
    assert_eq!(model.max_seq_length(), config.max_position_embeddings);

    assert!(
        !model.trainable_params().is_empty(),
        "target_modules=[Wqkv, Wo] must produce at least one trainable tensor",
    );

    // 3 sites (attn.Wqkv, attn.Wo, mlp.Wo) × 2 tensors per LoRA × num_layers.
    let expected = config.num_hidden_layers * 3 * 2;
    assert_eq!(
        model.trainable_params().len(),
        expected,
        "expected {expected} trainable tensors with target_modules=[Wqkv, Wo]",
    );
}
