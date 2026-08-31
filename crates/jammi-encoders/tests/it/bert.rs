//! BERT encoder integration tests against the `tiny_bert` fixture
//! (hidden_size=32, layers=1, heads=2, intermediate=128, max_pos=128).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_encoders::bert::BertConfig;
use jammi_encoders::{Bert, EncoderError, Pooling};
use jammi_lora::{FrozenBase, LoraBuildConfig, LoraInitMode, QuantizedLinear};

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
