//! BERT encoder integration tests against the `tiny_bert` fixture
//! (hidden_size=32, layers=1, heads=2, intermediate=128, max_pos=128).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarMap};
use jammi_encoders::bert::BertConfig;
use jammi_encoders::{Bert, EncoderError, Pooling};
use jammi_lora::{
    lora_linear_fused_dispatch_snapshot, lora_scaling, FrozenBase, LoraBuildConfig, LoraInitMode,
    QuantizedLinear,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

fn load_config() -> BertConfig {
    let config_path = fixture_dir().join("config.json");
    let raw = std::fs::read_to_string(&config_path).expect("read tiny_bert config.json");
    serde_json::from_str(&raw).expect("parse BertConfig")
}

fn weights_path() -> PathBuf {
    fixture_dir().join("model.safetensors")
}

#[test]
fn bert_loads_with_lora_frozen() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build frozen BERT on tiny_bert");

    assert!(
        bert.trainable_params().is_empty(),
        "frozen builder must produce zero trainable tensors"
    );
    assert_eq!(bert.hidden_size(), config.hidden_size);
    assert_eq!(bert.max_seq_length(), config.max_position_embeddings);
}

#[test]
fn bert_loads_with_target_modules() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let targets: Vec<String> = vec!["query".into(), "value".into()];
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

    let bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(lora)
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build LoRA-targeted BERT on tiny_bert");

    // 2 sites (q, v) × 2 tensors (lora_a, lora_b) per layer × num_hidden_layers.
    let expected = config.num_hidden_layers * 2 * 2;
    assert_eq!(
        bert.trainable_params().len(),
        expected,
        "expected {expected} trainable tensors with target_modules=[query, value]",
    );
}

/// Non-vacuity (#428 P2b round-2 fix): read `tiny_bert`'s OWN safetensors
/// header rather than merely stating in prose that its LoRA-eligible
/// sites carry a bias — every test in this file that claims to exercise
/// the bias-carrying fused site depends on this fixture fact staying
/// true, and a future fixture regeneration that dropped these biases
/// would otherwise leave those tests silently exercising the bias-FREE
/// path while still claiming to prove the bias-carrying one.
fn assert_tiny_bert_lora_sites_carry_a_bias(
    device: &Device,
    config: &BertConfig,
    weights: &std::path::Path,
) {
    let raw = candle_core::safetensors::load(weights, device)
        .expect("load tiny_bert safetensors for the bias non-vacuity check");
    for layer in 0..config.num_hidden_layers {
        for site in ["query", "value"] {
            let key = format!("encoder.layer.{layer}.attention.self.{site}.bias");
            assert!(
                raw.contains_key(&key),
                "tiny_bert fixture regressed: `{key}` must exist for this file's tests to \
                 actually exercise the bias-carrying fused site (non-vacuity)"
            );
        }
    }
}

fn build_bert_with_lora_on_biased_sites(
    device: &Device,
    config: &BertConfig,
    weights: &std::path::Path,
    varmap: &VarMap,
) -> Bert {
    // BERT's `query`/`value` sites (like every other linear this encoder
    // builds — `LoraSite::resolve_base`'s own `linear(..)` call) carry a
    // bias: #428 P2b's fused-with-bias pack applies here directly, not
    // the eager fallback every prior release forced. Non-vacuity checked
    // at runtime, not merely stated — see
    // `assert_tiny_bert_lora_sites_carry_a_bias`'s own doc.
    assert_tiny_bert_lora_sites_carry_a_bias(device, config, weights);
    let targets: Vec<String> = vec!["query".into(), "value".into()];
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
    Bert::builder()
        .pooling(Pooling::Mean)
        .lora(lora)
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights], config, device, varmap)
        .expect("build LoRA-targeted BERT on tiny_bert (biased sites)")
}

/// #428 P2b: BERT's LoRA-eligible sites (`query`/`value`, like every
/// other linear this encoder builds) carry a bias — the fused LoRA site
/// now FUSES a biased base instead of taking the eager fallback every
/// prior release forced (`base_has_no_bias`, DELETED). Counter-threading
/// (mirrors `modernbert.rs`'s own RoPE/softmax `set_training` gate
/// tests, `crate::modernbert::DISPATCH_COUNTER_TEST_LOCK` — this
/// integration-test binary's ONE process-wide dispatch-counter lock,
/// shared across every file that reads a fused-dispatch snapshot): a
/// training forward must dispatch the fused LoRA site (`fused` advances,
/// `eager` does not); an eval forward must touch NEITHER counter at all —
/// `LoraLinear::forward`'s own doc states eval never even reaches
/// `admit` (the load-bearing rule-9 assertion this test is: a byte
/// comparison alone would not catch eval accidentally routing through
/// `admit` with a domain-holds-but-training-false miswiring).
#[test]
fn bert_lora_bias_site_counter_threading_gates_the_fused_lora_linear_dispatch_counters() {
    let _guard = crate::modernbert::DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let mut bert = build_bert_with_lora_on_biased_sites(&device, &config, &weights, &varmap);
    // `LoraLinear::new_with_base` defaults `training: true` (a freshly
    // built LoRA-wrapped model is NOT eval-mode by default — unlike the
    // frozen-only models `modernbert.rs`'s own counter-threading tests
    // build, which have no `LoraLinear` at all to default anything on).
    // Establish the real eval baseline explicitly before measuring it.
    bert.set_training(false);
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1]], &device).unwrap();

    // Eval: forward must touch NEITHER dispatch counter.
    let before_eval = lora_linear_fused_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = lora_linear_fused_dispatch_snapshot();
    assert_eq!(
        (after_eval.fused, after_eval.eager),
        (before_eval.fused, before_eval.eager),
        "eval-mode forward must never touch the fused LoRA site's dispatch counters at all \
         (before={before_eval:?}, after={after_eval:?})"
    );

    // Training: 2 sites (query, value) per layer must each dispatch fused.
    bert.set_training(true);
    let before_train = lora_linear_fused_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused && after_train.eager == before_train.eager,
        "training-mode forward on a biased base must dispatch the fused LoRA site at least \
         once and never touch the eager counter (before={before_train:?}, after={after_train:?})"
    );

    // Back to eval: dispatch stops again.
    bert.set_training(false);
    let before_eval2 = lora_linear_fused_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward again");
    let after_eval2 = lora_linear_fused_dispatch_snapshot();
    assert_eq!(
        (after_eval2.fused, after_eval2.eager),
        (before_eval2.fused, before_eval2.eager),
        "set_training(false) must restore the eval-only path — neither counter advances \
         (before={before_eval2:?}, after={after_eval2:?})"
    );
}

/// Hand-composed eager reference for `tiny_bert`'s ONE encoder layer,
/// built entirely from plain `candle_nn`/`candle_core` PUBLIC ops plus the
/// fixture's own raw safetensors weights and the model's own (read-back,
/// not re-derived) LoRA `A`/`B` tensors — used only by
/// `bert_lora_bias_site_eval_matches_a_hand_composed_eager_reference_at_nonzero_ab`
/// below. Nothing here calls back into `jammi_lora`'s or
/// `jammi_encoders::bert`'s own forward machinery except the two PUBLIC
/// leaf primitives those modules themselves compose from
/// (`candle_nn::ops::layer_norm` — see `LayerNorm::forward`'s eval-mode
/// `(Some(bias), false)` arm in `jammi_encoders::layer_norm`, which
/// dispatches to this exact function; `jammi_encoders::contiguous_matmul`)
/// — neither takes a `LoraLinear`/`Bert` value as an argument, so nothing
/// about the LoRA dispatch decision this test exists to prove is
/// exercised a second time by calling them here. The additive attention
/// mask is deliberately NOT reconstructed: the test below uses an
/// all-valid mask (no padding), so `extended_attention_mask` would add
/// the zero tensor — omitting it is bit-exact against that, not an
/// approximation.
#[allow(clippy::too_many_arguments)]
fn hand_composed_reference_forward(
    device: &Device,
    config: &BertConfig,
    raw: &HashMap<String, Tensor>,
    lora_weights: &HashMap<String, Tensor>,
    scaling: f64,
    input_ids: &Tensor,
) -> Tensor {
    let get = |name: &str| -> Tensor {
        raw.get(name)
            .unwrap_or_else(|| panic!("tiny_bert fixture missing `{name}`"))
            .clone()
    };
    let hidden_size = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = hidden_size / heads;
    let (_batch, seq) = input_ids.dims2().unwrap();

    let word_emb = Embedding::new(get("embeddings.word_embeddings.weight"), hidden_size)
        .forward(input_ids)
        .unwrap();
    let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, device).unwrap();
    let token_type_emb =
        Embedding::new(get("embeddings.token_type_embeddings.weight"), hidden_size)
            .forward(&token_type_ids)
            .unwrap();
    let embeddings = (&word_emb + token_type_emb).unwrap();
    let position_ids = Tensor::arange(0u32, seq as u32, device).unwrap();
    let position_emb = Embedding::new(get("embeddings.position_embeddings.weight"), hidden_size)
        .forward(&position_ids)
        .unwrap();
    let embeddings = embeddings.broadcast_add(&position_emb).unwrap();
    let mut hidden = candle_nn::ops::layer_norm(
        &embeddings,
        &get("embeddings.LayerNorm.weight"),
        &get("embeddings.LayerNorm.bias"),
        config.layer_norm_eps as f32,
    )
    .unwrap();

    let transpose_for_scores = |x: &Tensor| -> Tensor {
        let mut new_shape = x.dims().to_vec();
        new_shape.pop();
        new_shape.push(heads);
        new_shape.push(head_dim);
        x.reshape(new_shape.as_slice())
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };

    for n in 0..config.num_hidden_layers {
        let p = format!("encoder.layer.{n}");
        let q_a = lora_weights
            .get(&format!("layer.{n}.query.lora_a"))
            .unwrap_or_else(|| panic!("missing layer.{n}.query.lora_a"));
        let q_b_lora = lora_weights
            .get(&format!("layer.{n}.query.lora_b"))
            .unwrap_or_else(|| panic!("missing layer.{n}.query.lora_b"));
        let v_a = lora_weights
            .get(&format!("layer.{n}.value.lora_a"))
            .unwrap_or_else(|| panic!("missing layer.{n}.value.lora_a"));
        let v_b_lora = lora_weights
            .get(&format!("layer.{n}.value.lora_b"))
            .unwrap_or_else(|| panic!("missing layer.{n}.value.lora_b"));

        let base_q = Linear::new(
            get(&format!("{p}.attention.self.query.weight")),
            Some(get(&format!("{p}.attention.self.query.bias"))),
        )
        .forward(&hidden)
        .unwrap();
        let lora_q_after_a = Linear::new(q_a.clone(), None).forward(&hidden).unwrap();
        let lora_q_out = Linear::new(q_b_lora.clone(), None)
            .forward(&lora_q_after_a)
            .unwrap();
        let q = (&base_q + &(&lora_q_out * scaling).unwrap()).unwrap();

        let k = Linear::new(
            get(&format!("{p}.attention.self.key.weight")),
            Some(get(&format!("{p}.attention.self.key.bias"))),
        )
        .forward(&hidden)
        .unwrap();

        let base_v = Linear::new(
            get(&format!("{p}.attention.self.value.weight")),
            Some(get(&format!("{p}.attention.self.value.bias"))),
        )
        .forward(&hidden)
        .unwrap();
        let lora_v_after_a = Linear::new(v_a.clone(), None).forward(&hidden).unwrap();
        let lora_v_out = Linear::new(v_b_lora.clone(), None)
            .forward(&lora_v_after_a)
            .unwrap();
        let v = (&base_v + &(&lora_v_out * scaling).unwrap()).unwrap();

        let q = transpose_for_scores(&q);
        let k = transpose_for_scores(&k);
        let v = transpose_for_scores(&v);

        let scores = jammi_encoders::contiguous_matmul(&q, &k.t().unwrap()).unwrap();
        let scores = (scores / (head_dim as f64).sqrt()).unwrap();
        let probs = candle_nn::ops::softmax(&scores, D::Minus1).unwrap();
        let context = jammi_encoders::contiguous_matmul(&probs, &v).unwrap();
        let context = context.transpose(1, 2).unwrap().contiguous().unwrap();
        let context = context.flatten_from(D::Minus2).unwrap();

        let attn_dense = Linear::new(
            get(&format!("{p}.attention.output.dense.weight")),
            Some(get(&format!("{p}.attention.output.dense.bias"))),
        )
        .forward(&context)
        .unwrap();
        let attn_out = candle_nn::ops::layer_norm(
            &(attn_dense + &hidden).unwrap(),
            &get(&format!("{p}.attention.output.LayerNorm.weight")),
            &get(&format!("{p}.attention.output.LayerNorm.bias")),
            config.layer_norm_eps as f32,
        )
        .unwrap();

        let inter = Linear::new(
            get(&format!("{p}.intermediate.dense.weight")),
            Some(get(&format!("{p}.intermediate.dense.bias"))),
        )
        .forward(&attn_out)
        .unwrap()
        .gelu_erf()
        .unwrap();
        let out_dense = Linear::new(
            get(&format!("{p}.output.dense.weight")),
            Some(get(&format!("{p}.output.dense.bias"))),
        )
        .forward(&inter)
        .unwrap();
        hidden = candle_nn::ops::layer_norm(
            &(out_dense + &attn_out).unwrap(),
            &get(&format!("{p}.output.LayerNorm.weight")),
            &get(&format!("{p}.output.LayerNorm.bias")),
            config.layer_norm_eps as f32,
        )
        .unwrap();
    }

    hidden
}

/// The eval-bytes half of rule 9 (a counter assertion alone is not
/// falsifiable against a numerically-silent regression) — K4 pin (#428
/// P2b round-2 fix): the PRIOR version of this test used
/// `LoraInitMode::ZerosB`, which makes LoRA's own contribution exactly
/// zero — comparing against a fully frozen model was TAUTOLOGICAL there
/// (it passed at main, before #428's fused-with-bias site existed at
/// all, unconditionally: eval never dispatches the fused site regardless,
/// and `B == 0` means the adapter adds nothing either way). `Gaussian`
/// init (seeded, deterministic) gives BOTH `A` and `B` non-zero values,
/// so this comparison genuinely exercises LoRA's own contribution: the
/// eval-mode output of a LoRA-wrapped, non-zero-`A`/`B` BERT must equal a
/// [`hand_composed_reference_forward`] built independently, from plain
/// tensor ops, off the SAME weights (base `Linear` with bias + `A`/`B` +
/// scaling, no dropout) — bitwise — after an explicit `is_finite` scan
/// over both sides (rule F: `NaN == NaN` is `false`, so a bare
/// `assert_eq!` on non-finite data can pass by both sides independently
/// producing NaN in the same positions without proving anything).
#[test]
fn bert_lora_bias_site_eval_matches_a_hand_composed_eager_reference_at_nonzero_ab() {
    let _guard = crate::modernbert::DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let config = load_config();
    let weights = weights_path();
    assert_tiny_bert_lora_sites_carry_a_bias(&device, &config, &weights);

    // Non-vacuity (round-2 fix, prove-bite finding): `tiny_bert`'s OWN
    // `query`/`value` biases are all EXACTLY zero (a fact of this
    // fixture, verified by hand-mutating the eager reference's own bias
    // argument to `None` and observing NO output change at all — the
    // fixture's real bias contributes nothing regardless). A bit-exact
    // comparison against a hand-composed reference would therefore stop
    // proving the bias's own contribution the moment the ONE differing
    // operand (`Some(bias)` vs a hypothetically dropped one) happened to
    // be zero everywhere — silently vacuous on the bias axis even though
    // `A`/`B` are genuinely non-zero. Patch in NON-zero, deterministic
    // bias values for every LoRA-eligible site, write the patched
    // tensors to a temp safetensors file, and build BOTH the production
    // model and the hand-composed reference off THAT file — never the
    // original, zero-bias one.
    let mut raw = candle_core::safetensors::load(&weights, &device)
        .expect("load tiny_bert's raw weights to patch in non-zero biases");
    for layer in 0..config.num_hidden_layers {
        for (site, phase) in [("query", 51i64), ("value", 52i64)] {
            let key = format!("encoder.layer.{layer}.attention.self.{site}.bias");
            let existing = raw
                .get(&key)
                .unwrap_or_else(|| panic!("tiny_bert fixture missing `{key}`"));
            let n = existing.dims1().unwrap();
            let patched_v: Vec<f32> = (0..n)
                .map(|i| (((i as i64 + phase + layer as i64) as f32) * 0.037).sin() * 0.5)
                .collect();
            assert!(
                patched_v.iter().any(|x| *x != 0.0),
                "the patched bias fixture must itself be non-zero"
            );
            let patched = Tensor::from_slice(&patched_v, n, &device).unwrap();
            raw.insert(key, patched);
        }
    }
    let patched_dir = tempfile::tempdir().expect("tempdir for the non-zero-bias fixture");
    let patched_path = patched_dir
        .path()
        .join("tiny_bert_nonzero_bias.safetensors");
    candle_core::safetensors::save(&raw, &patched_path)
        .expect("save the non-zero-bias-patched tiny_bert fixture");

    let varmap = VarMap::new();
    let targets: Vec<String> = vec!["query".into(), "value".into()];
    let no_layers: Option<Vec<usize>> = None;
    let empty_pattern: HashMap<String, usize> = HashMap::new();
    let lora_alpha = 8.0;
    let lora_rank = 4;
    let lora = LoraBuildConfig {
        target_modules: &targets,
        layers_to_transform: &no_layers,
        lora_rank,
        lora_alpha,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &empty_pattern,
        init_mode: LoraInitMode::Gaussian,
        seed: 7,
    };
    let mut bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(lora)
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[patched_path.as_path()], &config, &device, &varmap)
        .expect("build LoRA-targeted BERT on the non-zero-bias-patched tiny_bert (non-zero A/B)");
    // `LoraLinear::new_with_base` defaults `training: true` — an EVAL
    // comparison needs this explicit `false` (see the counter-threading
    // test's identical note).
    bert.set_training(false);

    // Every position is valid (no padding): `extended_attention_mask`
    // adds the zero tensor, so `hand_composed_reference_forward` (which
    // does not reconstruct that add at all) stays bit-exact against it.
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 1]], &device).unwrap();

    let before = lora_linear_fused_dispatch_snapshot();
    let eval_out = bert
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward on the non-zero-A/B model");
    let after = lora_linear_fused_dispatch_snapshot();
    assert_eq!(
        (after.fused, after.eager),
        (before.fused, before.eager),
        "eval-mode forward must never touch the fused LoRA site's dispatch counters at all, \
         even with non-zero A/B (before={before:?}, after={after:?})"
    );

    let lora_weights = bert
        .named_trainable_weights()
        .expect("read back the model's own A/B tensors for the reference");
    assert!(
        !lora_weights.is_empty(),
        "the model must actually carry LoRA A/B tensors for this comparison to be non-vacuous"
    );
    for (name, t) in &lora_weights {
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            v.iter().any(|x| *x != 0.0),
            "`{name}` must be non-zero (Gaussian init) — a zero A or B would make this test's \
             LoRA contribution vacuous again"
        );
    }
    // `raw` (built above, with the non-zero-patched biases) is reused
    // directly here — the reference must be built from the EXACT SAME
    // weights the production model above was built from, not the
    // original zero-bias fixture.
    let scaling = lora_scaling(lora_alpha, lora_rank, false).expect("compute LoRA scaling");
    let reference =
        hand_composed_reference_forward(&device, &config, &raw, &lora_weights, scaling, &input_ids);

    let eval_vec = eval_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let reference_vec = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        eval_vec.iter().all(|x| x.is_finite()),
        "the model's eval output must be entirely finite before it is trusted as a comparison \
         operand: {eval_vec:?}"
    );
    assert!(
        reference_vec.iter().all(|x| x.is_finite()),
        "the hand-composed reference output must be entirely finite before it is trusted as a \
         comparison operand: {reference_vec:?}"
    );
    assert_eq!(
        eval_vec, reference_vec,
        "eval output must be bit-identical to the hand-composed eager reference built from the \
         SAME weights (non-zero A/B), proving the bias-carrying fused-with-bias site's eval arm \
         still reproduces plain LoRA math exactly"
    );
}

#[test]
fn bert_forward_shape() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build BERT");

    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 0]], &device).unwrap();

    let pooled = bert.forward(&input_ids, &mask).expect("forward");
    assert_eq!(pooled.dims(), &[2, config.hidden_size]);

    let hidden = bert
        .forward_hidden(&input_ids, &mask)
        .expect("forward_hidden");
    assert_eq!(hidden.dims(), &[2, 5, config.hidden_size]);
}

#[test]
fn bert_pooling_variants() {
    let device = Device::Cpu;
    let config = load_config();
    let weights = weights_path();
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1]], &device).unwrap();

    let build_with = |p: Pooling| {
        let varmap = VarMap::new();
        Bert::builder()
            .pooling(p)
            .lora(LoraBuildConfig::frozen())
            .backbone_dtype(DType::F32)
            .adapter(None)
            .build(&[weights.as_path()], &config, &device, &varmap)
            .expect("build BERT for pooling variant")
    };

    let strategies = [
        Pooling::Mean,
        Pooling::Cls,
        Pooling::Max,
        Pooling::WeightedMean,
    ];
    let outputs: Vec<Vec<f32>> = strategies
        .iter()
        .map(|s| {
            build_with(*s)
                .forward(&input_ids, &mask)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .collect();

    // Pairwise distinct: every pair must differ in at least one coordinate
    // by more than the FP-noise floor.
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let max_diff = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff > 1e-4,
                "pooling {:?} and {:?} produced near-identical outputs (max |Δ| = {max_diff})",
                strategies[i],
                strategies[j]
            );
        }
    }
}

/// The additive attention mask is built in F32 (`mask.rs`); a F16 backbone
/// produces F16 attention scores, so the mask-add must cast to match or the
/// forward errors on a dtype mismatch. Real padding (not an all-ones mask)
/// is required to exercise a non-trivial additive mask.
#[test]
fn bert_forward_f16_backbone_with_padding() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F16)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build F16 BERT on tiny_bert");

    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 0, 0]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 0, 0]], &device).unwrap();

    let hidden = bert
        .forward_hidden(&input_ids, &mask)
        .expect("F16 forward_hidden with padding mask");
    assert_eq!(hidden.dims(), &[2, 5, config.hidden_size]);
    let values = hidden.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let values: Vec<f32> = values.to_vec1().unwrap();
    assert!(
        values.iter().all(|v| v.is_finite()),
        "F16 forward_hidden produced non-finite values"
    );
}

/// `forward` runs the pooled path (`forward_hidden` + `pool_and_normalize`).
/// The pooling mask is built in F32 (`pooling.rs`); a F16 backbone yields F16
/// hidden states, so every mask-combine site inside mean/max/weighted-mean
/// pooling must also cast to match. Real padding (not an all-ones mask) is
/// required to exercise a non-trivial mask in each strategy.
#[test]
fn bert_forward_pooled_f16_backbone_with_padding() {
    let device = Device::Cpu;
    let config = load_config();
    let weights = weights_path();

    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 0, 0]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 0, 0]], &device).unwrap();

    for strategy in [Pooling::Mean, Pooling::Max, Pooling::WeightedMean] {
        let varmap = VarMap::new();
        let bert = Bert::builder()
            .pooling(strategy)
            .lora(LoraBuildConfig::frozen())
            .backbone_dtype(DType::F16)
            .adapter(None)
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap_or_else(|e| panic!("build F16 BERT for {strategy:?} pooling: {e}"));

        let pooled = bert
            .forward(&input_ids, &mask)
            .unwrap_or_else(|e| panic!("F16 pooled forward with padding ({strategy:?}): {e}"));
        assert_eq!(pooled.dims(), &[2, config.hidden_size]);

        let values = pooled
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "F16 pooled forward ({strategy:?}) produced non-finite values"
        );
    }
}

/// An all-padding row (`attention_mask` entirely `0` for one batch item)
/// drives `mean_pool`'s and `weighted_mean_pool`'s divisor to `0` and removes
/// every real token from `max_pool`'s reduce. On base commit `4acad0f`, the
/// F32 eps floors (`1e-9` for the divisors, `-1e30` for the max-pool bias)
/// silently lose their guarantee once cast to F16: `1e-9` underflows to
/// `0.0` (giving `0 / 0 = NaN` for mean/weighted-mean) and `-1e30` overflows
/// to `-inf` (max-pool). This test is RED (NaN/-inf) on `4acad0f` and GREEN
/// on the fix, which floors the divisor at `1.0` (exact in every dtype) and
/// replaces the max-pool bias with a `where_cond` select against a
/// dtype-exact finite sentinel.
#[test]
fn bert_forward_pooled_f16_backbone_all_padding_row() {
    let device = Device::Cpu;
    let config = load_config();
    let weights = weights_path();

    // Row 0 is a normal, fully-real sequence. Row 1 is entirely padding.
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5], [0, 0, 0, 0, 0]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [0, 0, 0, 0, 0]], &device).unwrap();

    for strategy in [Pooling::Mean, Pooling::Max, Pooling::WeightedMean] {
        let varmap = VarMap::new();
        let bert = Bert::builder()
            .pooling(strategy)
            .lora(LoraBuildConfig::frozen())
            .backbone_dtype(DType::F16)
            .adapter(None)
            .build(&[weights.as_path()], &config, &device, &varmap)
            .unwrap_or_else(|e| panic!("build F16 BERT for {strategy:?} pooling: {e}"));

        let pooled = bert.forward(&input_ids, &mask).unwrap_or_else(|e| {
            panic!("F16 pooled forward with all-padding row ({strategy:?}): {e}")
        });
        assert_eq!(pooled.dims(), &[2, config.hidden_size]);

        let values = pooled
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "F16 pooled forward ({strategy:?}) produced non-finite values for an all-padding row: {values:?}"
        );

        // Row 1 (all-padding) must collapse to the exact zero vector for
        // mean/weighted-mean, not merely "some finite value".
        if matches!(strategy, Pooling::Mean | Pooling::WeightedMean) {
            let row1: Vec<f32> = pooled
                .narrow(0, 1, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                row1.iter().all(|v| v.abs() < 1e-6),
                "F16 all-padding row ({strategy:?}) expected the zero vector, got {row1:?}"
            );
        }
    }
}

#[test]
fn bert_max_seq_length_check() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build BERT");

    let seq = config.max_position_embeddings + 1;
    let row: Vec<u32> = vec![1; seq];
    let input_ids = Tensor::from_vec(row.clone(), (1, seq), &device).unwrap();
    let mask = Tensor::from_vec(row, (1, seq), &device).unwrap();

    match bert.forward_hidden(&input_ids, &mask) {
        Err(EncoderError::SequenceTooLong { seq: got, max }) => {
            assert_eq!(got, seq);
            assert_eq!(max, config.max_position_embeddings);
        }
        Err(other) => panic!("expected SequenceTooLong, got {other:?}"),
        Ok(_) => panic!("expected SequenceTooLong, got Ok"),
    }
}

/// A `weight_source` HIT whose quantized base geometry disagrees with
/// `config.json` (a GGUF sized for the wrong `in_features`, e.g. a checkpoint
/// built for a different `hidden_size`) must fail LOUDLY at `build()` — not
/// be accepted and only surface as a mismatched-shape matmul at first
/// `forward`. Every site queried returns the SAME wrong-shaped quantized
/// base unconditionally, so the very first `LoraSite::resolve_base` call
/// (layer 0's `query` projection) is guaranteed to hit the mismatch.
#[test]
fn bert_rejects_a_weight_source_hit_with_mismatched_geometry() {
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    // tiny_bert's hidden_size is 32; every attention/FFN linear at this
    // fixture is square (h -> h) or (h -> intermediate) — a base sized
    // in_features=64 (double the true hidden_size, and a multiple of
    // GgmlDType::Q8_0's 32-element block size) disagrees with EVERY call
    // site's expected shape.
    let wrong_in_features = config.hidden_size * 2;
    let wrong_out_features = config.hidden_size;
    let w_v: Vec<f32> = (0..wrong_out_features * wrong_in_features)
        .map(|i| ((i as f64) * 0.037 + 0.3).sin() as f32)
        .collect();
    let w = Tensor::from_vec(w_v, (wrong_out_features, wrong_in_features), &device).unwrap();
    let wrong_weight = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap());

    // The lookup returns the SAME wrong-shaped quantized weight for every
    // queried name (an `Arc` clone, cheap and shareable across the `Fn`
    // trait object's repeated calls) — every site the builder queries hits
    // the same mismatch.
    let lookup = move |_name: &str| -> Result<Option<FrozenBase>, EncoderError> {
        Ok(Some(FrozenBase::Quantized(
            QuantizedLinear::new(wrong_weight.clone(), None).unwrap(),
        )))
    };

    // `Bert` does not implement `Debug`, so `Result::expect_err` (which
    // requires `T: Debug` to format the panic message on an unexpected
    // `Ok`) is not usable here — match explicitly instead.
    let result = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .weight_source(&lookup)
        .build(&[weights.as_path()], &config, &device, &varmap);
    let err = match result {
        Ok(_) => panic!("wrong-shaped weight_source hit must fail at build(), not at forward()"),
        Err(e) => e,
    };

    match err {
        EncoderError::Config(msg) => {
            assert!(
                msg.contains("geometry mismatch"),
                "expected a geometry-mismatch message, got: {msg}"
            );
            assert!(
                msg.contains(&wrong_in_features.to_string())
                    && msg.contains(&config.hidden_size.to_string()),
                "expected the message to name expected/actual shapes, got: {msg}"
            );
        }
        other => panic!("expected EncoderError::Config, got {other:?}"),
    }
}

/// #460 (C-LN): before this unit, EVERY BERT LayerNorm carried a bias, so
/// a real BERT training run's `ln` dispatch-counter pair read `0/0`
/// regardless of how many LayerNorms it actually ran (no `admit()` call
/// site at all — see `jammi_kernels::ops::layer_norm`'s module doc). This
/// pins the fix, end to end, on a real (non-LoRA) BERT build: training
/// must dispatch the fused `layer_norm_fused` key at least once, eval
/// must never touch it at all.
#[test]
fn bert_biased_layer_norm_counter_threading_gates_the_ln_dispatch_counters() {
    let _guard = crate::modernbert::DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let config = load_config();
    let varmap = VarMap::new();
    let weights = weights_path();

    let mut bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights.as_path()], &config, &device, &varmap)
        .expect("build BERT");

    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1]], &device).unwrap();

    // Eval: every BERT LayerNorm is biased, so `forward`'s first arm
    // (`candle_nn::ops::layer_norm`) runs directly -- the `ln` counter
    // pair must not move at all.
    bert.set_training(false);
    let before_eval = jammi_encoders::ln_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = jammi_encoders::ln_dispatch_snapshot();
    assert_eq!(
        (after_eval.fused, after_eval.eager),
        (before_eval.fused, before_eval.eager),
        "eval-mode forward must never touch the `ln` dispatch counters at all \
         (before={before_eval:?}, after={after_eval:?})"
    );

    // Training: every biased LayerNorm in this fixture is F32, contiguous,
    // well within MAX_HIDDEN -- the fused biased kernel's domain holds, so
    // the fused counter must advance (this is the exact `0/0` bug #460
    // fixes: before it, this assertion would fail with fused == before).
    bert.set_training(true);
    let before_train = jammi_encoders::ln_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = jammi_encoders::ln_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused && after_train.eager == before_train.eager,
        "training-mode forward on an all-biased BERT must dispatch the fused LayerNorm \
         kernel at least once and never fall back to the eager path \
         (before={before_train:?}, after={after_train:?})"
    );

    // Back to eval: dispatch stops again.
    bert.set_training(false);
    let before_eval2 = jammi_encoders::ln_dispatch_snapshot();
    let _ = bert
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward again");
    let after_eval2 = jammi_encoders::ln_dispatch_snapshot();
    assert_eq!(
        (after_eval2.fused, after_eval2.eager),
        (before_eval2.fused, before_eval2.eager),
        "set_training(false) must restore the eval-only path -- neither counter advances \
         (before={before_eval2:?}, after={after_eval2:?})"
    );
}
