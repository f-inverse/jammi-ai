//! BERT encoder integration tests against the `tiny_bert` fixture
//! (hidden_size=32, layers=1, heads=2, intermediate=128, max_pos=128).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_encoders::bert::BertConfig;
use jammi_encoders::{Bert, EncoderError, Pooling};
use jammi_lora::{LoraBuildConfig, LoraInitMode};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_bert")
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
