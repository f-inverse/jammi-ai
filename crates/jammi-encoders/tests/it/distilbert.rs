//! DistilBERT integration tests. CPU-only, hermetic.
//!
//! No HuggingFace fixture is available for DistilBERT in this crate, so we
//! synthesise a minimal safetensors archive of randomly-initialised weights
//! and exercise the API surface against it. Numerical parity against
//! candle-transformers is covered by the BERT parity test.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarMap};
use jammi_encoders::distilbert::DistilBertConfig;
use jammi_encoders::{DistilBert, EncoderError, Pooling};
use jammi_lora::{
    lora_linear_fused_dispatch_snapshot, lora_scaling, FrozenBase, LoraBuildConfig, LoraInitMode,
    QuantizedLinear,
};
use tempfile::tempdir;

/// Hand-rolled synthetic config matching the spec's exit criteria for test 6.
fn tiny_config() -> jammi_encoders::distilbert::DistilBertConfig {
    serde_json::from_value(serde_json::json!({
        "dim": 32,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_dim": 64,
        "vocab_size": 100,
        "max_position_embeddings": 128,
    }))
    .expect("synthetic config deserialises")
}

/// Construct a random F32 safetensors archive containing every key the
/// DistilBert builder requires for the supplied config, returning the file
/// path and the owning [`tempfile::TempDir`] (kept alive by the caller).
fn write_synthetic_weights(
    config: &jammi_encoders::distilbert::DistilBertConfig,
    device: &Device,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("distilbert.safetensors");

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;
    let max_pos = config.max_position_embeddings;

    let randn = |shape: (usize, usize)| -> Tensor {
        Tensor::randn(0f32, 0.02, shape, device).expect("randn 2-D")
    };
    let randn_1d =
        |size: usize| -> Tensor { Tensor::randn(0f32, 0.02, (size,), device).expect("randn 1-D") };
    let ones_1d =
        |size: usize| -> Tensor { Tensor::ones((size,), DType::F32, device).expect("ones 1-D") };
    let zeros_1d =
        |size: usize| -> Tensor { Tensor::zeros((size,), DType::F32, device).expect("zeros 1-D") };

    // Embeddings
    tensors.insert(
        "distilbert.embeddings.word_embeddings.weight".into(),
        randn((vocab, h)),
    );
    tensors.insert(
        "distilbert.embeddings.position_embeddings.weight".into(),
        randn((max_pos, h)),
    );
    tensors.insert("distilbert.embeddings.LayerNorm.weight".into(), ones_1d(h));
    tensors.insert("distilbert.embeddings.LayerNorm.bias".into(), zeros_1d(h));

    for n in 0..config.num_hidden_layers {
        let prefix = format!("distilbert.transformer.layer.{n}");

        for lin in ["q_lin", "k_lin", "v_lin", "out_lin"] {
            tensors.insert(format!("{prefix}.attention.{lin}.weight"), randn((h, h)));
            tensors.insert(format!("{prefix}.attention.{lin}.bias"), randn_1d(h));
        }
        tensors.insert(format!("{prefix}.sa_layer_norm.weight"), ones_1d(h));
        tensors.insert(format!("{prefix}.sa_layer_norm.bias"), zeros_1d(h));

        tensors.insert(format!("{prefix}.ffn.lin1.weight"), randn((inter, h)));
        tensors.insert(format!("{prefix}.ffn.lin1.bias"), randn_1d(inter));
        tensors.insert(format!("{prefix}.ffn.lin2.weight"), randn((h, inter)));
        tensors.insert(format!("{prefix}.ffn.lin2.bias"), randn_1d(h));

        tensors.insert(format!("{prefix}.output_layer_norm.weight"), ones_1d(h));
        tensors.insert(format!("{prefix}.output_layer_norm.bias"), zeros_1d(h));
    }

    candle_core::safetensors::save(&tensors, &path).expect("save safetensors");
    (dir, path)
}

#[test]
fn distilbert_loads_and_forwards() {
    let device = Device::Cpu;
    let config = tiny_config();
    let (_dir, weights_path) = write_synthetic_weights(&config, &device);

    let varmap = VarMap::new();
    let encoder = DistilBert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights_path.as_path()], &config, &device, &varmap)
        .expect("builder succeeds on synthetic weights");

    assert_eq!(encoder.hidden_size(), config.hidden_size);
    assert_eq!(encoder.max_seq_length(), config.max_position_embeddings);

    let input_ids =
        Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 0]], &device).expect("input_ids tensor");
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 0]], &device).expect("mask tensor");

    let pooled = encoder
        .forward(&input_ids, &mask)
        .expect("pooled forward succeeds");
    assert_eq!(pooled.dims(), &[2, config.hidden_size]);

    let hidden = encoder
        .forward_hidden(&input_ids, &mask)
        .expect("hidden forward succeeds");
    assert_eq!(hidden.dims(), &[2, 5, config.hidden_size]);

    assert!(
        encoder.trainable_params().is_empty(),
        "frozen LoRA config must yield zero trainable params, got {}",
        encoder.trainable_params().len()
    );
}

/// A `weight_source` HIT whose quantized base geometry disagrees with
/// `config.json` (a GGUF sized for the wrong `dim`) must fail LOUDLY at
/// `build()` — not be accepted and only surface as a mismatched-shape
/// matmul at first `forward`. Every site queried returns the SAME
/// wrong-shaped quantized base unconditionally, so the very first
/// `LoraSlot::build_in` call (layer 0's `q_lin`) is guaranteed to hit the
/// mismatch.
#[test]
fn distilbert_rejects_a_weight_source_hit_with_mismatched_geometry() {
    let device = Device::Cpu;
    let config = tiny_config();
    let (_dir, weights_path) = write_synthetic_weights(&config, &device);
    let varmap = VarMap::new();

    // tiny_config's `dim` (hidden_size) is 32; every attention linear at
    // this fixture is square (h -> h) — a base sized in_features=64
    // (double the true hidden_size, and a multiple of GgmlDType::Q8_0's
    // 32-element block size) disagrees with EVERY call site's expected
    // shape.
    let wrong_in_features = config.hidden_size * 2;
    let wrong_out_features = config.hidden_size;
    let w_v: Vec<f32> = (0..wrong_out_features * wrong_in_features)
        .map(|i| ((i as f64) * 0.037 + 0.3).sin() as f32)
        .collect();
    let w = Tensor::from_vec(w_v, (wrong_out_features, wrong_in_features), &device).unwrap();
    let wrong_weight = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap());

    let lookup = move |_name: &str| -> Result<Option<FrozenBase>, EncoderError> {
        Ok(Some(FrozenBase::Quantized(
            QuantizedLinear::new(wrong_weight.clone(), None).unwrap(),
        )))
    };

    // `DistilBert` does not implement `Debug`, so `Result::expect_err`
    // (which requires `T: Debug` to format the panic message on an
    // unexpected `Ok`) is not usable here — match explicitly instead.
    let result = DistilBert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .weight_source(&lookup)
        .build(&[weights_path.as_path()], &config, &device, &varmap);
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

fn build_distilbert_with_lora_on_biased_sites(
    device: &Device,
    config: &DistilBertConfig,
    weights: &std::path::Path,
    varmap: &VarMap,
    init_mode: LoraInitMode,
    seed: u64,
) -> DistilBert {
    // `write_synthetic_weights` writes a `randn_1d` bias for every
    // attention/FFN linear (this file's own module doc: "synthesise a
    // minimal safetensors archive... containing every key the builder
    // requires") — DistilBERT's `q_lin`/`v_lin` sites carry a bias here
    // exactly like BERT's `query`/`value` (see `bert.rs`'s
    // `build_bert_with_lora_on_biased_sites`'s identical note): #428
    // P2b's fused-with-bias pack applies here directly.
    let targets: Vec<String> = vec!["q_lin".into(), "v_lin".into()];
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
        init_mode,
        seed,
    };
    DistilBert::builder()
        .pooling(Pooling::Mean)
        .lora(lora)
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights], config, device, varmap)
        .expect("build LoRA-targeted DistilBert on the synthetic biased fixture")
}

/// #428 P2b: mirrors `bert.rs`'s
/// `bert_lora_bias_site_counter_threading_gates_the_fused_lora_linear_dispatch_counters`
/// for DistilBERT's biased `q_lin`/`v_lin` sites — same
/// process-wide dispatch-counter lock
/// (`crate::modernbert::DISPATCH_COUNTER_TEST_LOCK`), same both-counter
/// pairing (a fused-dispatching arm must ALSO prove the eager counter did
/// not move, and vice versa; an eval forward must touch NEITHER at all).
#[test]
fn distilbert_lora_bias_site_counter_threading_gates_the_fused_lora_linear_dispatch_counters() {
    let _guard = crate::modernbert::DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let config = tiny_config();
    let (_dir, weights_path) = write_synthetic_weights(&config, &device);
    let varmap = VarMap::new();

    let mut encoder = build_distilbert_with_lora_on_biased_sites(
        &device,
        &config,
        &weights_path,
        &varmap,
        LoraInitMode::ZerosB,
        0,
    );
    // `LoraLinear::new_with_base` defaults `training: true` — establish
    // the eval baseline explicitly before measuring it (see `bert.rs`'s
    // identical note).
    encoder.set_training(false);
    let input_ids =
        Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 0]], &device).expect("input_ids tensor");
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 0]], &device).expect("mask tensor");

    let before_eval = lora_linear_fused_dispatch_snapshot();
    let _ = encoder
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward");
    let after_eval = lora_linear_fused_dispatch_snapshot();
    assert_eq!(
        (after_eval.fused, after_eval.eager),
        (before_eval.fused, before_eval.eager),
        "eval-mode forward must never touch the fused LoRA site's dispatch counters at all \
         (before={before_eval:?}, after={after_eval:?})"
    );

    encoder.set_training(true);
    let before_train = lora_linear_fused_dispatch_snapshot();
    let _ = encoder
        .forward_hidden(&input_ids, &mask)
        .expect("training forward");
    let after_train = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_train.fused > before_train.fused && after_train.eager == before_train.eager,
        "training-mode forward on a biased base must dispatch the fused LoRA site at least \
         once and never touch the eager counter (before={before_train:?}, after={after_train:?})"
    );

    encoder.set_training(false);
    let before_eval2 = lora_linear_fused_dispatch_snapshot();
    let _ = encoder
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

/// Hand-composed eager reference for the synthetic DistilBERT fixture's
/// ONE transformer block, built entirely from plain
/// `candle_nn`/`candle_core` PUBLIC ops plus the fixture's own raw
/// safetensors weights and the model's own (read-back, not re-derived)
/// LoRA `A`/`B` tensors — mirrors `bert.rs`'s
/// `hand_composed_reference_forward`; see that function's doc for the
/// non-tautology argument (same PUBLIC-leaf-primitives-only construction:
/// `candle_nn::ops::layer_norm`, `jammi_encoders::contiguous_matmul`).
/// Adapted for DistilBERT's architecture: no `token_type_embeddings`,
/// POST-LN residual order (`LayerNorm(x + sublayer(x))`), `q_lin`/`k_lin`/
/// `v_lin`/`out_lin` attention naming, `lin1`/`lin2` FFN naming. The
/// additive attention mask is likewise NOT reconstructed — the test below
/// uses an all-valid mask (no padding), so `extended_attention_mask`
/// would add the zero tensor.
#[allow(clippy::too_many_arguments)]
fn hand_composed_reference_forward(
    device: &Device,
    config: &DistilBertConfig,
    raw: &HashMap<String, Tensor>,
    lora_weights: &HashMap<String, Tensor>,
    scaling: f64,
    input_ids: &Tensor,
) -> Tensor {
    let get = |name: &str| -> Tensor {
        raw.get(name)
            .unwrap_or_else(|| panic!("synthetic fixture missing `{name}`"))
            .clone()
    };
    let hidden_size = config.hidden_size;
    let heads = config.num_attention_heads;
    let head_dim = hidden_size / heads;
    let (_batch, seq) = input_ids.dims2().unwrap();

    let word_emb = Embedding::new(
        get("distilbert.embeddings.word_embeddings.weight"),
        hidden_size,
    )
    .forward(input_ids)
    .unwrap();
    let position_ids = Tensor::arange(0u32, seq as u32, device).unwrap();
    let position_emb = Embedding::new(
        get("distilbert.embeddings.position_embeddings.weight"),
        hidden_size,
    )
    .forward(&position_ids)
    .unwrap();
    let embeddings = word_emb.broadcast_add(&position_emb).unwrap();
    let mut hidden = candle_nn::ops::layer_norm(
        &embeddings,
        &get("distilbert.embeddings.LayerNorm.weight"),
        &get("distilbert.embeddings.LayerNorm.bias"),
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
        let p = format!("distilbert.transformer.layer.{n}");
        let q_a = lora_weights
            .get(&format!("layer.{n}.q_lin.lora_a"))
            .unwrap_or_else(|| panic!("missing layer.{n}.q_lin.lora_a"));
        let q_b_lora = lora_weights
            .get(&format!("layer.{n}.q_lin.lora_b"))
            .unwrap_or_else(|| panic!("missing layer.{n}.q_lin.lora_b"));
        let v_a = lora_weights
            .get(&format!("layer.{n}.v_lin.lora_a"))
            .unwrap_or_else(|| panic!("missing layer.{n}.v_lin.lora_a"));
        let v_b_lora = lora_weights
            .get(&format!("layer.{n}.v_lin.lora_b"))
            .unwrap_or_else(|| panic!("missing layer.{n}.v_lin.lora_b"));

        let base_q = Linear::new(
            get(&format!("{p}.attention.q_lin.weight")),
            Some(get(&format!("{p}.attention.q_lin.bias"))),
        )
        .forward(&hidden)
        .unwrap();
        let lora_q_after_a = Linear::new(q_a.clone(), None).forward(&hidden).unwrap();
        let lora_q_out = Linear::new(q_b_lora.clone(), None)
            .forward(&lora_q_after_a)
            .unwrap();
        let q = (&base_q + &(&lora_q_out * scaling).unwrap()).unwrap();

        let k = Linear::new(
            get(&format!("{p}.attention.k_lin.weight")),
            Some(get(&format!("{p}.attention.k_lin.bias"))),
        )
        .forward(&hidden)
        .unwrap();

        let base_v = Linear::new(
            get(&format!("{p}.attention.v_lin.weight")),
            Some(get(&format!("{p}.attention.v_lin.bias"))),
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

        let attn_out = Linear::new(
            get(&format!("{p}.attention.out_lin.weight")),
            Some(get(&format!("{p}.attention.out_lin.bias"))),
        )
        .forward(&context)
        .unwrap();
        let attn_residual = (&attn_out + &hidden).unwrap();
        let attn_normed = candle_nn::ops::layer_norm(
            &attn_residual,
            &get(&format!("{p}.sa_layer_norm.weight")),
            &get(&format!("{p}.sa_layer_norm.bias")),
            config.layer_norm_eps as f32,
        )
        .unwrap();

        let mid = Linear::new(
            get(&format!("{p}.ffn.lin1.weight")),
            Some(get(&format!("{p}.ffn.lin1.bias"))),
        )
        .forward(&attn_normed)
        .unwrap();
        let activated = mid.gelu_erf().unwrap();
        let ffn_out = Linear::new(
            get(&format!("{p}.ffn.lin2.weight")),
            Some(get(&format!("{p}.ffn.lin2.bias"))),
        )
        .forward(&activated)
        .unwrap();
        let ffn_residual = (&ffn_out + &attn_normed).unwrap();
        hidden = candle_nn::ops::layer_norm(
            &ffn_residual,
            &get(&format!("{p}.output_layer_norm.weight")),
            &get(&format!("{p}.output_layer_norm.bias")),
            config.layer_norm_eps as f32,
        )
        .unwrap();
    }

    hidden
}

/// K4 pin (#428 P2b), mirrors `bert.rs`'s
/// `bert_lora_bias_site_eval_matches_a_hand_composed_eager_reference_at_nonzero_ab` —
/// see that test's own doc for why `LoraInitMode::Gaussian` (non-zero `A`
/// AND `B`) rather than the tautological `ZerosB` is required for this
/// comparison to be non-vacuous. Eval must (a) touch NEITHER dispatch
/// counter and (b) reproduce [`hand_composed_reference_forward`]
/// bit-for-bit, after an explicit `is_finite` scan over both sides.
#[test]
fn distilbert_lora_bias_site_eval_matches_a_hand_composed_eager_reference_at_nonzero_ab() {
    let _guard = crate::modernbert::DISPATCH_COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let device = Device::Cpu;
    let config = tiny_config();
    let (_dir, weights_path) = write_synthetic_weights(&config, &device);
    let varmap = VarMap::new();

    let lora_alpha = 8.0;
    let lora_rank = 4;
    let mut encoder = build_distilbert_with_lora_on_biased_sites(
        &device,
        &config,
        &weights_path,
        &varmap,
        LoraInitMode::Gaussian,
        11,
    );
    encoder.set_training(false);

    let input_ids =
        Tensor::new(&[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device).expect("input_ids tensor");
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 1]], &device).expect("mask tensor");

    let before = lora_linear_fused_dispatch_snapshot();
    let eval_out = encoder
        .forward_hidden(&input_ids, &mask)
        .expect("eval forward on the non-zero-A/B model");
    let after = lora_linear_fused_dispatch_snapshot();
    assert_eq!(
        (after.fused, after.eager),
        (before.fused, before.eager),
        "eval-mode forward must never touch the fused LoRA site's dispatch counters at all, \
         even with non-zero A/B (before={before:?}, after={after:?})"
    );

    let lora_weights = encoder
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
    let raw = candle_core::safetensors::load(&weights_path, &device)
        .expect("load the synthetic fixture's raw weights for the hand-composed reference");
    // Non-vacuity, checked at runtime rather than assumed from
    // `write_synthetic_weights`'s own `randn_1d` call (round-2 fix: see
    // `bert.rs`'s sibling test's doc for why this crate does not trust a
    // fixture's bias to be non-zero without checking — `tiny_bert`'s OWN
    // real bias values are all exactly zero, which would have made an
    // analogous check there vacuously pass).
    for n in 0..config.num_hidden_layers {
        for site in ["q_lin", "v_lin"] {
            let key = format!("distilbert.transformer.layer.{n}.attention.{site}.bias");
            let bias_t = raw
                .get(&key)
                .unwrap_or_else(|| panic!("synthetic fixture missing `{key}`"));
            let bias_v = bias_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(
                bias_v.iter().any(|x| *x != 0.0),
                "`{key}` must be non-zero for this bias-carrying comparison to be non-vacuous"
            );
        }
    }
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
