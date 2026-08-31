//! DistilBERT integration tests. CPU-only, hermetic.
//!
//! No HuggingFace fixture is available for DistilBERT in this crate, so we
//! synthesise a minimal safetensors archive of randomly-initialised weights
//! and exercise the API surface against it. Numerical parity against
//! candle-transformers is covered by the BERT parity test.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_encoders::{DistilBert, EncoderError, Pooling};
use jammi_lora::{FrozenBase, LoraBuildConfig, QuantizedLinear};
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
