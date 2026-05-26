//! Numerical parity test: assert that `jammi_encoders::Bert` (frozen, no LoRA)
//! reproduces `candle_transformers::models::bert::BertModel` byte-for-byte
//! within F32 noise tolerance.
//!
//! Gated behind the `parity-test` feature so the default `cargo test` stays
//! free of any candle-transformers runtime dependency.
#![cfg(feature = "parity-test")]

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use jammi_encoders::{Bert, BertConfig, Pooling};
use jammi_lora::LoraBuildConfig;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

#[test]
fn frozen_jammi_bert_matches_candle_transformers() {
    let device = Device::Cpu;
    let config_path = fixture_dir().join("config.json");
    let weights_path = fixture_dir().join("model.safetensors");
    let config_str = std::fs::read_to_string(&config_path).expect("read tiny_bert config.json");

    let jammi_config: BertConfig =
        serde_json::from_str(&config_str).expect("parse jammi BertConfig");
    let varmap = VarMap::new();
    let jammi_bert = Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights_path.as_path()], &jammi_config, &device, &varmap)
        .expect("build jammi BERT");

    let ref_config: candle_transformers::models::bert::Config =
        serde_json::from_str(&config_str).expect("parse candle-transformers Config");
    let ref_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.as_path()], DType::F32, &device)
            .expect("mmap reference VarBuilder")
    };
    let ref_bert = candle_transformers::models::bert::BertModel::load(ref_vb, &ref_config)
        .expect("load candle-transformers BertModel");

    let input_ids = Tensor::new(&[[2u32, 121, 124, 1, 3], [2, 121, 1, 3, 0]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 0]], &device).unwrap();
    let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, &device).unwrap();

    let jammi_hidden = jammi_bert
        .forward_hidden(&input_ids, &mask)
        .expect("jammi forward_hidden");
    let ref_hidden = ref_bert
        .forward(&input_ids, &token_type_ids, Some(&mask))
        .expect("reference forward");

    assert_eq!(
        jammi_hidden.dims(),
        ref_hidden.dims(),
        "jammi vs candle-transformers shape mismatch"
    );

    let diff = (&jammi_hidden - &ref_hidden).unwrap().abs().unwrap();
    let max_diff: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        max_diff < 1e-5,
        "jammi Bert output differs from candle-transformers by max |Δ| = {max_diff}"
    );
}
