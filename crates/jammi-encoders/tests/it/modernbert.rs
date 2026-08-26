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

/// Relative-ULP multiplier for the fold-order tolerance below. Measured
/// (a100c pod, `crates/jammi-kernels/src/admission.rs` gate at `86aa9da`):
/// row1's worst per-element hidden-state ratio was **220.93** `f32::EPSILON`
/// units relative to `max(|padded|,|unpadded|)` (2 layers of LN/Wqkv-GEMM/
/// attention/Wo-GEMM/GeGLU compound a batch-of-2-vs-batch-of-1 GEMM
/// blocking difference well past a "couple of ULP" bound). `512` gives
/// roughly 2.3x headroom over that measured worst case while staying a
/// small multiple of `f32::EPSILON`, not an unbounded fudge.
const FOLD_ORDER_RTOL_ULP: f32 = 512.0;
/// Absolute-tolerance multiplier (also in `f32::EPSILON` units), needed
/// IN ADDITION to the relative term above because LoRA `B` is
/// zero-initialized: `dL/dA = Bᵀ·upstream = 0` exactly (both legs agree,
/// ratio 0), but `dL/dB` for a short, near-cancelling batch=1 sequence
/// can itself be numerical noise clustered around a true value of ~0 —
/// `max(|padded|,|unpadded|,f32::EPSILON)` then floors the RELATIVE
/// denominator at `f32::EPSILON` (a unit sized for values near 1.0, not a
/// sensible floor for a tensor whose own true magnitude is far smaller),
/// which inflates the ULP-ratio metric without the term being large in
/// any absolute sense. Measured worst case: `layer.0.Wo.lora_b`'s padded
/// grad differs from the summed unpadded legs' own (near-zero) value by
/// **1.91e-6** while every element of both unpadded legs' OWN tensor
/// stays within `f32::EPSILON` of exactly zero. `64 * f32::EPSILON` ≈
/// `7.63e-6` gives ~4x headroom over that measured residual — the SAME
/// `64` this file's original grad tolerance used as its (insufficient
/// alone) relative multiplier, reused here as the atol term so the two
/// constants in this file share one convention instead of introducing an
/// unrelated third number.
const FOLD_ORDER_ATOL_ULP: f32 = 64.0;

/// Standard `atol + rtol * scale` combined bound (the numpy/torch
/// `allclose` form), in `f32::EPSILON` units on both terms — a pure
/// relative bound (as this file originally used) is not sufficient when
/// the REFERENCE value is itself near-exactly zero (see
/// `FOLD_ORDER_ATOL_ULP`'s doc); a pure absolute bound would hide real
/// divergence at large-magnitude elements (the guide's §3.8 "no absolute
/// ULP floor" clause) — the combined form avoids both failure modes.
fn fold_order_bound(scale: f32) -> f32 {
    f32::EPSILON * (FOLD_ORDER_ATOL_ULP + FOLD_ORDER_RTOL_ULP * scale)
}

/// Elementwise oracle + the shared scale a linear (Higham) summation-error
/// bound derives from: for every element of `unpadded`'s real row against
/// the CORRESPONDING row (`batch_idx`) of `padded`'s real positions
/// (`0..len`), asserts `|padded − unpadded| <= fold_order_bound(scale)`
/// where `scale = max(|padded|, |unpadded|)` — the larger of the two
/// PRE-cancellation values (see the oracle's own doc above) — and returns
/// `(Σ scale, n)` over the compared elements, the exact quantities the
/// loss bound below is built from (a sum of `n` terms each within
/// `fold_order_bound(scale_i)` of their target is within
/// `Σ fold_order_bound(scale_i) = atol_ulp * EPS * n + rtol_ulp * EPS *
/// Σscale_i` of the target sum, by linearity).
fn assert_real_row_matches_and_scale(
    padded: &Tensor,
    batch_idx: usize,
    unpadded: &Tensor,
    len: usize,
    label: &str,
) -> (f32, usize) {
    let p_row: Vec<f32> = padded
        .narrow(0, batch_idx, 1)
        .unwrap()
        .narrow(1, 0, len)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let u_row: Vec<f32> = unpadded
        .narrow(0, 0, 1)
        .unwrap()
        .narrow(1, 0, len)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(
        p_row.len(),
        u_row.len(),
        "{label}: padded and unpadded real rows must cover the same element count"
    );
    let mut scale_sum = 0.0f32;
    for (i, (&p, &u)) in p_row.iter().zip(u_row.iter()).enumerate() {
        assert!(
            p.is_finite() && u.is_finite(),
            "{label}[{i}]: non-finite hidden state (padded={p}, unpadded={u})"
        );
        let scale = p.abs().max(u.abs());
        let bound = fold_order_bound(scale);
        let diff = (p - u).abs();
        assert!(
            diff.is_finite() && diff <= bound,
            "{label}[{i}]: padded hidden state must match the unpadded row's own value within \
             the derived fold-order tolerance -- batch-size-dependent GEMM blocking/accumulation \
             order, not a batch-mixing leak (padded={p}, unpadded={u}, diff={diff}, bound={bound})"
        );
        scale_sum += scale;
    }
    (scale_sum, p_row.len())
}

/// THE oracle: a batch with real padding (row 0 fully real, row 1 padded)
/// vs the SAME two rows run UNPADDED one-by-one — f32 on CPU (the block
/// arm — `AttentionBlockFused` admits on this fixture, contract v4 §1 F1).
///
/// **Finding, not the deliverable's original assumption (widened on the
/// a100c pod, `P6 B3-dense`, `crates/jammi-kernels/src/admission.rs`
/// commit `86aa9da`'s gate run):** the LOSS is NOT bit-identical between
/// the two legs on every host. The original assumption ("the forward pass
/// is row-independent, so `hidden_padded[0, s, :]` for `s < 6` is
/// literally the same computation as `hidden_row0[s, :]`") is true
/// MATHEMATICALLY but not at the FLOAT level: candle's CPU `gemm` picks
/// its blocking/accumulation order from the operand shape, and a
/// batch-of-2 GEMM (`M=2` rows total) is not guaranteed to issue the SAME
/// per-row reduction order as two independent batch-of-1 GEMMs (`M=1`
/// each) — floating-point addition is non-associative (family J), so a
/// different fold order over mathematically-identical terms is not
/// guaranteed bit-identical. This box hits it (macOS's `gemm` blocking
/// happened not to depend on `M` at this tiny shape; this pod's does);
/// x86_64/Linux CPU BLAS microkernel selection differing by total-M is
/// architecture-dependent, not a code defect.
///
/// Rather than fudge a single ad hoc constant on the near-zero (post-
/// LayerNorm, heavily cancelled) final scalar loss, the SAME per-element
/// few-ULP tolerance already established below for the LoRA gradients is
/// asserted one level EARLIER, directly on the compared hidden-state
/// elements (`assert_real_row_matches_and_scale`), and the loss bound is
/// then a linear CONSEQUENCE of that per-element bound (Higham's
/// summation-error bound: if every one of `n` summed terms differs by at
/// most `k_i`, the sum differs by at most `Σk_i`) — one derived constant
/// (`64 * f32::EPSILON` per element, relative to the larger of the two
/// pre-cancellation values, never the cancelled result) drives both the
/// loss and the gradient tolerance, not two independently-chosen fudge
/// factors. Pad rows still contribute EXACTLY nothing (the non-vacuity
/// control below proves the reduction itself excludes pad columns; the
/// bound above would need to be many orders of magnitude looser than a
/// few ULPs if a pad row's residual value were leaking in — see
/// `padded_training_oracle_is_non_vacuous_including_pad_columns_changes_the_loss`).
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

    // Elementwise oracle FIRST (proves row-independence at the tensor
    // level, within a derived fold-order tolerance) -- its returned scale
    // sums and element counts are what the loss bound below is a linear
    // consequence of, per this test's own doc comment.
    let (scale_sum_row0, n_row0) =
        assert_real_row_matches_and_scale(&hidden_padded, 0, &hidden_row0, 6, "row0");
    let (scale_sum_row1, n_row1) =
        assert_real_row_matches_and_scale(&hidden_padded, 1, &hidden_row1, 4, "row1");

    let loss_summed = loss_row0_v + loss_row1_v;
    let loss_diff = (loss_padded_v - loss_summed).abs();
    // Σ fold_order_bound(scale_i) = atol_ulp*EPS*n + rtol_ulp*EPS*Σscale_i,
    // by linearity of the per-element bound established above.
    let loss_bound = f32::EPSILON
        * (FOLD_ORDER_ATOL_ULP * (n_row0 + n_row1) as f32
            + FOLD_ORDER_RTOL_ULP * (scale_sum_row0 + scale_sum_row1));
    assert!(
        loss_diff.is_finite() && loss_diff <= loss_bound,
        "the padded batch's real-rows loss must match the sum of the two unpadded rows' own \
         losses within the SAME derived fold-order tolerance summed over every compared \
         element, not bit-identical -- batch-size-dependent GEMM rounding, not a batch-mixing \
         leak (padded={loss_padded_v}, row0={loss_row0_v}, row1={loss_row1_v}, \
         summed={loss_summed}, diff={loss_diff}, bound={loss_bound})"
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
            // set of terms is not guaranteed bit-identical. `B` is
            // zero-initialized, so `dL/dA = Bᵀ·upstream = 0` exactly for
            // both legs (the `scale`-floor case below never fires there),
            // while `dL/dB` for a short batch=1 sequence can itself be
            // near-zero numerical noise -- `fold_order_bound` (this file's
            // module-level doc) is the SAME `atol + rtol*scale` bound
            // `assert_real_row_matches_and_scale` uses above, not a second
            // independently-chosen constant.
            let scale = g_row0[i].abs().max(g_row1[i].abs());
            let bound = fold_order_bound(scale);
            assert!(
                diff.is_finite() && diff <= bound,
                "{name}[{i}]: padded grad must match the sum of the two unpadded rows' own \
                 grads within the derived fold-order tolerance (row0={}, row1={}, padded={}, \
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
