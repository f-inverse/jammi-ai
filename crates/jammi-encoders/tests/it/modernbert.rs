//! ModernBERT encoder integration tests against the `tiny_modernbert_classifier`
//! fixture (hidden_size=32, layers=1, heads=2, intermediate=64, max_pos=128).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
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
        seed: 0,
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

/// The gate test the C3 audit asked for: `ModernBert::set_training`'s
/// threading of `training` down to `ModernBertAttention` (and from there
/// to `RotaryEmbedding::apply_training`) is exercised through the REAL
/// encoder's forward call graph, not just `RotaryEmbedding` in isolation.
/// This is the ONLY test in this repository that would catch an
/// accidental deletion of the `layer.attention.set_training(training)`
/// line inside `ModernBert::set_training` — every other RoPE test either
/// constructs a bare `RotaryEmbedding` directly (bypassing
/// `ModernBertAttention`/`ModernBert` entirely) or never toggles
/// `set_training` at all.
///
/// Race-safety note: `jammi_encoders::rope_dispatch_snapshot()` reads a
/// PROCESS-WIDE static shared by every test in this binary. Exact
/// before/after equality (used below for the eval legs) would be racy
/// if another test in this same integration-test binary concurrently
/// exercised training-mode RoPE — traced at the time of writing, no
/// other test in `tests/it/` calls `set_training` or builds anything
/// other than a frozen/eval-mode model, so this test is the sole toucher
/// of that counter here. A future test added alongside this one that
/// also drives `training = true` would need to switch these to the
/// monotonic (`>=`) form `crate::layer_norm`'s own shared-static tests use.
#[test]
fn set_training_threading_gates_the_fused_rope_dispatch_counters() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let mut model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build ModernBert on tiny_modernbert_classifier");

    let input_ids = Tensor::new(&[[2u32, 5, 10, 3]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1]], &device).unwrap();

    // Eval (the model's default state): forward must NOT dispatch the
    // fused RoPE kernel at all.
    let before_eval = jammi_encoders::rope_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = jammi_encoders::rope_dispatch_snapshot();
    assert_eq!(
        after_eval.fused, before_eval.fused,
        "eval-mode forward must never dispatch the fused RoPE kernel \
         (before={before_eval:?}, after={after_eval:?})"
    );

    // Training: this fixture's head_dim (hidden_size=32 / num_heads=2 ==
    // 16) is fused-eligible on CPU, so a training forward MUST advance
    // the fused counter — a regression here (e.g. `set_training`'s
    // threading line deleted) would silently leave this at zero forever.
    model.set_training(true);
    let before_train = jammi_encoders::rope_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = jammi_encoders::rope_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused,
        "training-mode forward with set_training(true) must dispatch the fused \
         RoPE kernel at least once — this is the exact regression the C3 audit's \
         'this is the only test that would catch deletion of the set_training \
         threading' note names (before={before_train:?}, after={after_train:?})"
    );

    // Back to eval: the fused dispatch path must stop again.
    model.set_training(false);
    let before_eval2 = jammi_encoders::rope_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward again");
    let after_eval2 = jammi_encoders::rope_dispatch_snapshot();
    assert_eq!(
        after_eval2.fused, before_eval2.fused,
        "set_training(false) must restore the eval-only dispatch path \
         (before={before_eval2:?}, after={after_eval2:?})"
    );
}
