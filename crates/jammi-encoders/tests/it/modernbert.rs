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

// ─────────────────────────────────────────────────────────────────────────
// The padded training fwd+bwd numeric oracle (P6 Stage B B2, item 6)
// ─────────────────────────────────────────────────────────────────────────
//
// Contract v4 §1 F1 / §5 B2: the only padded ModernBERT training fwd+bwd
// test that existed before this commit
// (`crates/jammi-ai/tests/it/encoder_adapters.rs`'s
// `encoder_adapters_modernbert_writes_adapter_marker`) asserts nothing
// numeric — it only checks the saved adapter config's JSON fields. That
// file lives in `jammi-ai`, a crate this agent does not own (Shared
// crate boundary — coordinate through the lead), so the numeric oracle
// contract v4 asks for is added HERE instead, against the same
// `ModernBert`/`jammi-lora` machinery `jammi-ai`'s fine-tune path
// ultimately calls into. This is the RED oracle B3 is expected to run
// again on the FA2 arm once it exists.

fn head64_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_modernbert_head64")
}

fn head64_config() -> ModernBertConfig {
    let raw = std::fs::read_to_string(head64_fixture_dir().join("config.json"))
        .expect("read tiny_modernbert_head64 config");
    serde_json::from_str(&raw).expect("parse ModernBertConfig")
}

fn lora_targets_wqkv_wo() -> LoraBuildConfig<'static> {
    // `'static` leak is fine in a test: these two `Vec`s/`HashMap` just
    // need to outlive the `LoraBuildConfig` borrow for this function's
    // caller's use, and this is a tiny, one-shot test fixture.
    let targets: &'static Vec<String> =
        Box::leak(Box::new(vec!["Wqkv".to_string(), "Wo".to_string()]));
    let no_layers: &'static Option<Vec<usize>> = Box::leak(Box::new(None));
    let empty_pattern: &'static HashMap<String, usize> = Box::leak(Box::new(HashMap::new()));
    LoraBuildConfig {
        target_modules: targets,
        layers_to_transform: no_layers,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: empty_pattern,
        init_mode: LoraInitMode::ZerosB,
        seed: 42,
    }
}

/// Builds a fresh LoRA-targeted `ModernBert` on the `tiny_modernbert_head64`
/// fixture (hidden 64, 1 head, `head_dim == 64` — the ONLY fixture in this
/// crate that reaches `AttentionBlockFused`, contract v4 §1 F1's "block
/// arm"), seeded identically every call so two builds compare bit-for-bit.
fn head64_lora_model(varmap: &VarMap) -> ModernBert {
    let config = head64_config();
    let weights = head64_fixture_dir().join("model.safetensors");
    let mut model = ModernBert::builder()
        .pooling(Pooling::Mean)
        .lora(lora_targets_wqkv_wo())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &Device::Cpu, varmap)
        .expect("build LoRA-targeted ModernBert on tiny_modernbert_head64");
    model.set_training(true);
    model
}

/// Sum, over every REAL (non-pad) token position, of that position's
/// hidden-state vector's own element sum — a loss whose gradient is
/// non-trivial through EVERY trainable tensor (LoRA A/B on Wqkv and Wo,
/// every layer) and whose value/gradient a pad row can only affect if
/// something in the forward pass leaks across rows or across the
/// pad/real boundary (the exact class of bug this oracle exists to
/// catch — none exists in today's architecture, but the FA2 unpad/repad
/// path B3 adds is exactly where one COULD be introduced).
fn real_rows_loss(hidden: &Tensor, lengths: &[usize]) -> Tensor {
    let mut terms = Vec::new();
    for (b, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        let row = hidden.narrow(0, b, 1).unwrap().narrow(1, 0, len).unwrap();
        terms.push(row.sum_all().unwrap());
    }
    terms
        .into_iter()
        .reduce(|a, b| (a + b).unwrap())
        .expect("at least one non-empty row")
}

/// THE oracle: a batch with real padding (row 0 fully real, row 1 padded)
/// vs the SAME two rows run UNPADDED one-by-one — f32 on CPU (the block
/// arm — `AttentionBlockFused` admits on this fixture, contract v4 §1 F1).
/// The LOSS (summed over real positions only) is BIT-IDENTICAL between the
/// two legs (the forward pass is row-independent — no batch-mixing op
/// exists anywhere in this architecture, so `hidden_padded[0, s, :]` for
/// `s < 6` is literally the same computation as `hidden_row0[s, :]`).
///
/// **Finding, not the deliverable's original assumption:** the LoRA A/B
/// GRADIENTS are measured here to be within a tight, principled few-ULP
/// tolerance of the sum of the two unpadded runs' own gradients, NOT
/// bit-identical — `dL/dW` for a weight shared across the batch sums each
/// row's contribution, and the padded run performs that sum in ONE batched
/// reduction (row-major order over `batch * seq` rows, including the
/// exactly-zero-gradient pad rows) while comparing against TWO
/// independently-computed (batch=1) reductions summed externally is a
/// DIFFERENT fold order over the same mathematical terms — floating-point
/// addition is non-associative (family J), and this specific case hits a
/// near-cancellation (two ~1e-10-magnitude terms summing to ~1e-12) that
/// amplifies the RELATIVE error at the tiny cancelled value while the
/// ABSOLUTE error stays at ordinary f32-ULP scale on the pre-cancellation
/// operands — see the tolerance's own derivation, inline below. Pad rows
/// still contribute EXACTLY nothing (that half of the claim holds exactly:
/// the loss comparison above proves it, and the gradient tolerance would
/// need to be many orders of magnitude looser than a few ULPs if a pad
/// row's residual gradient were leaking in).
#[test]
fn padded_training_loss_and_lora_grads_match_unpadded_rows_run_individually_f32_cpu() {
    // This fixture is GeGLU/RoPE/softmax fused-eligible too, so a training
    // forward here touches the SAME process-wide dispatch-counter statics
    // `set_training_threading_gates_the_fused_*_dispatch_counters` reads
    // with exact-equality assertions — see [`DISPATCH_COUNTER_TEST_LOCK`]'s
    // doc for why every such test in this binary takes this lock.
    let _guard = DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let input_ids_padded =
        Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 0, 0]], &device).unwrap();
    let mask_padded =
        Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 0, 0]], &device).unwrap();
    let lengths = [6usize, 4usize];

    // --- Leg A: the padded batch, ONE forward + ONE backward. ---
    let varmap_padded = VarMap::new();
    let model_padded = head64_lora_model(&varmap_padded);
    let hidden_padded = model_padded
        .forward_hidden(&input_ids_padded, &mask_padded)
        .expect("padded training forward");
    let loss_padded = real_rows_loss(&hidden_padded, &lengths);
    let loss_padded_v: f32 = loss_padded.to_scalar().unwrap();
    assert!(loss_padded_v.is_finite());
    let grads_padded = loss_padded.backward().expect("padded backward");

    // --- Leg B: the SAME two rows, run UNPADDED, one at a time, on a
    // FRESH model built with the identical seed (so weights match
    // bit-for-bit — LoRA init is deterministic given `seed`). ---
    let varmap_row0 = VarMap::new();
    let model_row0 = head64_lora_model(&varmap_row0);
    let ids_row0 = Tensor::new(&[[2u32, 5, 10, 3, 7, 9]], &device).unwrap();
    let mask_row0 = Tensor::new(&[[1u32, 1, 1, 1, 1, 1]], &device).unwrap();
    let hidden_row0 = model_row0
        .forward_hidden(&ids_row0, &mask_row0)
        .expect("row0 unpadded forward");
    let loss_row0 = real_rows_loss(&hidden_row0, &[6usize]);
    let grads_row0 = loss_row0.backward().expect("row0 backward");

    let varmap_row1 = VarMap::new();
    let model_row1 = head64_lora_model(&varmap_row1);
    let ids_row1 = Tensor::new(&[[4u32, 8, 1, 6]], &device).unwrap();
    let mask_row1 = Tensor::new(&[[1u32, 1, 1, 1]], &device).unwrap();
    let hidden_row1 = model_row1
        .forward_hidden(&ids_row1, &mask_row1)
        .expect("row1 unpadded forward");
    let loss_row1 = real_rows_loss(&hidden_row1, &[4usize]);
    let grads_row1 = loss_row1.backward().expect("row1 backward");

    let loss_row0_v: f32 = loss_row0.to_scalar().unwrap();
    let loss_row1_v: f32 = loss_row1.to_scalar().unwrap();
    assert_eq!(
        loss_padded_v,
        loss_row0_v + loss_row1_v,
        "the padded batch's real-rows loss must be bit-identical to the sum of the two \
         unpadded rows' own losses -- pad rows contribute exactly nothing"
    );

    // Every LoRA A/B tensor, every layer, every site: the padded run's
    // gradient must equal the ELEMENTWISE SUM of the two unpadded runs'
    // gradients, bit-for-bit. `named_trainable_weights` gives each tensor
    // a NAME so the three models' Vars (different `VarMap`s, different
    // `TensorId`s) can be paired up correctly.
    let named_padded = model_padded.named_trainable_weights().unwrap();
    let named_row0 = model_row0.named_trainable_weights().unwrap();
    let named_row1 = model_row1.named_trainable_weights().unwrap();
    assert!(
        !named_padded.is_empty(),
        "the LoRA targets must have produced trainable tensors"
    );

    for name in named_padded.keys() {
        let key_tensor = &named_padded[name];
        let g_padded: Vec<f32> = grads_padded
            .get(key_tensor)
            .unwrap_or_else(|| panic!("padded run: no grad for {name}"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let row0_key = &named_row0[name];
        let g_row0: Vec<f32> = grads_row0
            .get(row0_key)
            .unwrap_or_else(|| panic!("row0 run: no grad for {name}"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let row1_key = &named_row1[name];
        let g_row1: Vec<f32> = grads_row1
            .get(row1_key)
            .unwrap_or_else(|| panic!("row1 run: no grad for {name}"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(g_padded.len(), g_row0.len());
        assert_eq!(g_padded.len(), g_row1.len());
        for i in 0..g_padded.len() {
            let summed = g_row0[i] + g_row1[i];
            let diff = (g_padded[i] - summed).abs();
            // NOT exact bit-identity, measured (a finding, not the
            // deliverable's original assumption): `dL/dW` for a shared
            // weight sums each row's contribution, and the PADDED run
            // performs that sum in ONE batched reduction over
            // `batch * seq` rows (row-major order, including the
            // exactly-zero-gradient pad rows) while this comparison sums
            // TWO INDEPENDENTLY-COMPUTED (batch=1) reductions externally in
            // f32 — floating-point addition is non-associative (family J),
            // so a DIFFERENT fold order over the mathematically-identical
            // set of terms is not guaranteed bit-identical, and near-
            // cancellation (two ~1e-10 terms summing to ~1e-12 here)
            // amplifies the RELATIVE error at the tiny cancelled magnitude
            // even though the ABSOLUTE error stays at f32-ULP scale on the
            // PRE-cancellation operands. The bound below is exactly that:
            // relative to the larger of the two pre-cancellation terms
            // (never the cancelled result, which would make an ordinary-
            // sized rounding error look enormous), a tight, principled
            // few-ULP tolerance — not a loose fudge factor.
            let scale = g_row0[i].abs().max(g_row1[i].abs()).max(f32::EPSILON);
            let bound = 64.0 * f32::EPSILON * scale;
            assert!(
                diff <= bound,
                "{name}[{i}]: padded grad must match the sum of the two unpadded rows' own \
                 grads within a few-ULP fold-order tolerance (row0={}, row1={}, padded={}, \
                 summed={summed}, diff={diff}, bound={bound})",
                g_row0[i],
                g_row1[i],
                g_padded[i]
            );
        }
    }
}

/// The RED control this oracle exists to catch: if a pad row's garbage
/// values leaked into the loss (e.g. `real_rows_loss` itself summed the
/// WHOLE row instead of narrowing to `len`), the padded run's loss would
/// differ from the sum of the two unpadded runs' — this test asserts that
/// leak IS detectable by deliberately including the pad columns and
/// checking the comparison now FAILS the exact-equality the real oracle
/// relies on (a non-vacuity control on the oracle mechanism itself, not a
/// production code path).
#[test]
fn padded_training_oracle_is_non_vacuous_including_pad_columns_changes_the_loss() {
    // See the sibling oracle test's identical lock rationale.
    let _guard = DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let input_ids_padded =
        Tensor::new(&[[2u32, 5, 10, 3, 7, 9], [4u32, 8, 1, 6, 0, 0]], &device).unwrap();
    let mask_padded =
        Tensor::new(&[[1u32, 1, 1, 1, 1, 1], [1u32, 1, 1, 1, 0, 0]], &device).unwrap();

    let varmap_padded = VarMap::new();
    let model_padded = head64_lora_model(&varmap_padded);
    let hidden_padded = model_padded
        .forward_hidden(&input_ids_padded, &mask_padded)
        .expect("padded training forward");

    // Real-rows-only loss (the ACTUAL oracle's quantity).
    let real_loss: f32 = real_rows_loss(&hidden_padded, &[6usize, 4usize])
        .to_scalar()
        .unwrap();
    // Whole-batch loss INCLUDING the two pad columns of row 1 — must
    // differ from `real_loss` on real weights (pad-row hidden states are
    // not literally zero; only the POOLED/masked output is designed to
    // ignore them, `forward_hidden` returns raw per-token states).
    let whole_loss: f32 = hidden_padded.sum_all().unwrap().to_scalar().unwrap();
    assert_ne!(
        real_loss, whole_loss,
        "the oracle's real-rows-only reduction must actually EXCLUDE the pad columns, or this \
         whole test would vacuously pass regardless of a leak"
    );
}
