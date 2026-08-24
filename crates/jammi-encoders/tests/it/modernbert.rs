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

/// Serializes every test in this file that reads the process-wide fused-
/// dispatch counters (`rope_dispatch_snapshot` / `softmax_dispatch_snapshot`)
/// around a `set_training` toggle, so an "eval must not advance the
/// counter" exact-equality assertion in one such test cannot be made
/// flaky by another such test's training-mode forward incrementing the
/// SAME shared static concurrently (`cargo test`'s default per-binary
/// thread pool runs `#[test]` fns in parallel). This does not serialize
/// the whole file — only the handful of tests that actually toggle
/// `set_training` and read a dispatch snapshot take the lock; every other
/// test in this binary is unaffected. `pub(crate)`: `tests/it/modernbert_sliding_window.rs`'s
/// training-mode fixture tests also drive `set_training(true)` and take
/// this SAME lock (`crate::modernbert::DISPATCH_COUNTER_TEST_LOCK`) — one
/// lock shared across every file in this integration-test binary that
/// touches the process-wide dispatch counters, not one per file.
pub(crate) static DISPATCH_COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
/// before/after equality (used below for the eval legs) would be racy if
/// another test in this same integration-test binary concurrently
/// exercised training-mode RoPE — this test and
/// `set_training_threading_gates_the_fused_softmax_dispatch_counters`
/// (below) both do exactly that, so both take
/// [`DISPATCH_COUNTER_TEST_LOCK`] for their duration, serializing just the
/// two of them against each other (every other test in this binary is
/// unaffected).
#[test]
fn set_training_threading_gates_the_fused_rope_dispatch_counters() {
    let _guard = DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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

/// The C4 (fused masked softmax) equivalent of the gate test above:
/// `ModernBert::set_training`'s threading down to `ModernBertAttention`
/// (and from there to `softmax_apply_training`) exercised through the REAL
/// encoder's forward call graph. See
/// [`set_training_threading_gates_the_fused_rope_dispatch_counters`]'s doc
/// for the race-safety rationale behind [`DISPATCH_COUNTER_TEST_LOCK`],
/// which this test also takes.
#[test]
fn set_training_threading_gates_the_fused_softmax_dispatch_counters() {
    let _guard = DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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
    // fused softmax kernel at all.
    let before_eval = jammi_encoders::softmax_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = jammi_encoders::softmax_dispatch_snapshot();
    assert_eq!(
        after_eval.fused, before_eval.fused,
        "eval-mode forward must never dispatch the fused softmax kernel \
         (before={before_eval:?}, after={after_eval:?})"
    );

    // Training: this fixture's scores shape ([batch=1, heads=2, seq=4,
    // seq=4], rank 4, last=4) is fused-eligible on CPU, so a training
    // forward MUST advance the fused counter -- a regression here (e.g.
    // the `self.training` gate in `ModernBertAttention::forward` deleted)
    // would silently leave this at zero forever.
    model.set_training(true);
    let before_train = jammi_encoders::softmax_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = jammi_encoders::softmax_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused,
        "training-mode forward with set_training(true) must dispatch the fused \
         softmax kernel at least once (before={before_train:?}, after={after_train:?})"
    );

    // Back to eval: the fused dispatch path must stop again.
    model.set_training(false);
    let before_eval2 = jammi_encoders::softmax_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward again");
    let after_eval2 = jammi_encoders::softmax_dispatch_snapshot();
    assert_eq!(
        after_eval2.fused, before_eval2.fused,
        "set_training(false) must restore the eval-only dispatch path \
         (before={before_eval2:?}, after={after_eval2:?})"
    );
}

/// Same real, end-to-end wiring proof as
/// `set_training_threading_gates_the_fused_rope_dispatch_counters` /
/// `..._softmax_dispatch_counters`, for the C5 fused GeGLU kernel (see
/// `set_training_threading_gates_the_fused_rope_dispatch_counters`'s doc
/// for the race-safety rationale behind [`DISPATCH_COUNTER_TEST_LOCK`],
/// which this test also takes).
#[test]
fn set_training_threading_gates_the_fused_geglu_dispatch_counters() {
    let _guard = DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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
    // fused GeGLU kernel at all.
    let before_eval = jammi_encoders::geglu_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = jammi_encoders::geglu_dispatch_snapshot();
    assert_eq!(
        after_eval.fused, before_eval.fused,
        "eval-mode forward must never dispatch the fused GeGLU kernel \
         (before={before_eval:?}, after={after_eval:?})"
    );

    // Training: this fixture's `intermediate_size` (64, even and nonzero)
    // is fused-eligible on CPU, so a training forward MUST advance the
    // fused counter -- a regression here (e.g. the `self.training` gate
    // in `ModernBertMlp::forward` or its `set_training` threading line
    // deleted) would silently leave this at zero forever.
    model.set_training(true);
    let before_train = jammi_encoders::geglu_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = jammi_encoders::geglu_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused,
        "training-mode forward with set_training(true) must dispatch the fused \
         GeGLU kernel at least once (before={before_train:?}, after={after_train:?})"
    );

    // Back to eval: the fused dispatch path must stop again.
    model.set_training(false);
    let before_eval2 = jammi_encoders::geglu_dispatch_snapshot();
    let _ = model
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward again");
    let after_eval2 = jammi_encoders::geglu_dispatch_snapshot();
    assert_eq!(
        after_eval2.fused, before_eval2.fused,
        "set_training(false) must restore the eval-only dispatch path \
         (before={before_eval2:?}, after={after_eval2:?})"
    );
}
