//! Integration tests for the `jammi-lora` public surface. CPU-only, hermetic.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{
    effective_rank, load_adapter, save_adapter, should_apply_lora, AdapterConfig, BackboneDtype,
    LoraBuildConfig, LoraInitMode, LoraLinear, MaybeLoraLinear,
};

fn cpu() -> Device {
    Device::Cpu
}

fn build_base(in_features: usize, out_features: usize, device: &Device) -> Linear {
    // Deterministic non-zero base so we can compare base(x) against the
    // identity-at-init LoRA forward.
    let mut row = Vec::with_capacity(in_features * out_features);
    for i in 0..out_features {
        for j in 0..in_features {
            row.push(((i * 7 + j * 3) as f32).sin());
        }
    }
    let w = Tensor::from_vec(row, (out_features, in_features), device).unwrap();
    Linear::new(w, None)
}

fn rand_input(device: &Device) -> Tensor {
    Tensor::randn(0f32, 1.0, (2, 5, 8), device).unwrap()
}

#[test]
fn lora_linear_zeros_b_init_is_identity() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base = build_base(8, 16, &device);
    let x = rand_input(&device);

    let base_out = base.forward(&x).unwrap();
    let lora = LoraLinear::new(base, 4, 8.0, false, LoraInitMode::ZerosB, None, &vb).unwrap();
    let lora_out = lora.forward(&x).unwrap();

    let diff = (&lora_out - &base_out).unwrap().abs().unwrap();
    let max: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(
        max < 1e-6,
        "ZerosB init should be identity, got max |Δ| = {max}"
    );
}

#[test]
fn lora_linear_gaussian_init_diverges_from_base() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base = build_base(8, 16, &device);
    let x = rand_input(&device);

    let base_out = base.forward(&x).unwrap();
    let lora = LoraLinear::new(base, 4, 8.0, false, LoraInitMode::Gaussian, None, &vb).unwrap();
    let lora_out = lora.forward(&x).unwrap();

    let diff = (&lora_out - &base_out).unwrap().abs().unwrap();
    let max: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(max > 1e-6, "Gaussian init should diverge from base");
}

#[test]
fn lora_linear_trainable_params_count() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base = build_base(8, 16, &device);
    let lora = LoraLinear::new_simple(base, 4, 8.0, &vb).unwrap();
    assert_eq!(lora.trainable_params().len(), 2);
}

#[test]
fn lora_linear_rslora_scaling() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base = build_base(8, 16, &device);
    let x = rand_input(&device);

    // Build an adapter with non-zero B so scaling is observable: load tensors
    // by hand via `from_loaded`, then re-check scaling math by comparing
    // RSLoRA against vanilla.
    let lora_a = Tensor::ones((4, 8), DType::F32, &device).unwrap();
    let lora_b = Tensor::ones((16, 4), DType::F32, &device).unwrap();

    // alpha = 4.0, rank = 4 — vanilla scaling = 1.0, RSLoRA = 2.0.
    let vanilla = LoraLinear::from_loaded(
        build_base(8, 16, &device),
        lora_a.clone(),
        lora_b.clone(),
        4.0,
    );
    // For RSLoRA we have to construct via `new` to exercise the use_rslora
    // path; we then overwrite A/B with ones so the delta is observable.
    let mut rslora = LoraLinear::new(base, 4, 4.0, true, LoraInitMode::ZerosB, None, &vb).unwrap();
    rslora.lora_a = lora_a;
    rslora.lora_b = lora_b;

    let v_out = vanilla.forward(&x).unwrap();
    let r_out = rslora.forward(&x).unwrap();

    // Recover the deltas relative to base and assert the ratio is 2.0.
    let base_out = build_base(8, 16, &device).forward(&x).unwrap();
    let v_delta = (&v_out - &base_out).unwrap();
    let r_delta = (&r_out - &base_out).unwrap();

    let v_norm: f32 = v_delta
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_scalar()
        .unwrap();
    let r_norm: f32 = r_delta
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_scalar()
        .unwrap();

    let ratio = r_norm / v_norm;
    assert!(
        (ratio - 2.0).abs() < 1e-4,
        "RSLoRA scaling expected 2x vanilla, got ratio {ratio}"
    );
}

#[test]
fn maybe_lora_linear_frozen_forward_matches_underlying() {
    let device = cpu();
    let base = build_base(8, 16, &device);
    let frozen = MaybeLoraLinear::Frozen(build_base(8, 16, &device));

    let x = rand_input(&device);
    let direct = base.forward(&x).unwrap();
    let wrapped = frozen.forward(&x).unwrap();

    let diff = (&direct - &wrapped).unwrap().abs().unwrap();
    let max: f32 = diff
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar()
        .unwrap();
    assert!(max < 1e-6, "Frozen wrapper differs from underlying Linear");
}

#[test]
fn maybe_lora_linear_named_weights_only_for_lora() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let frozen = MaybeLoraLinear::Frozen(build_base(8, 16, &device));
    assert!(frozen.named_weights("query").unwrap().is_empty());

    let lora = MaybeLoraLinear::Lora(
        LoraLinear::new_simple(build_base(8, 16, &device), 4, 8.0, &vb).unwrap(),
    );
    let weights = lora.named_weights("query").unwrap();
    assert_eq!(weights.len(), 2);
    assert!(weights.contains_key("query.lora_a"));
    assert!(weights.contains_key("query.lora_b"));
}

#[test]
fn should_apply_lora_exact_match() {
    let targets = vec!["query".to_string()];
    assert!(should_apply_lora("query", &targets, 0, &None));
    assert!(should_apply_lora(
        "attention.self.query",
        &targets,
        0,
        &None
    ));
    assert!(!should_apply_lora("key", &targets, 0, &None));
    assert!(!should_apply_lora("queryless", &targets, 0, &None));
}

#[test]
fn should_apply_lora_all_linear() {
    let targets = vec!["all-linear".to_string()];
    assert!(should_apply_lora("query", &targets, 0, &None));
    assert!(should_apply_lora("anything.at.all", &targets, 7, &None));
}

#[test]
fn should_apply_lora_layer_filter() {
    let targets = vec!["all-linear".to_string()];
    let layers = Some(vec![2usize, 5]);
    assert!(should_apply_lora("query", &targets, 2, &layers));
    assert!(should_apply_lora("query", &targets, 5, &layers));
    assert!(!should_apply_lora("query", &targets, 0, &layers));
    assert!(!should_apply_lora("query", &targets, 3, &layers));
}

#[test]
fn effective_rank_substring_match() {
    let mut pattern = HashMap::new();
    pattern.insert("query".to_string(), 16);
    assert_eq!(effective_rank("query", 8, &pattern), 16);
    assert_eq!(
        effective_rank("attention.self.query.dense", 8, &pattern),
        16
    );
    assert_eq!(effective_rank("key", 8, &pattern), 8);
}

#[test]
fn adapter_config_json_roundtrip() {
    let mut rank_pattern = HashMap::new();
    rank_pattern.insert("query".to_string(), 16);
    let cfg = AdapterConfig {
        model_type: "bert".into(),
        lora_rank: 8,
        lora_alpha: 16.0,
        use_rslora: true,
        target_modules: vec!["query".into(), "value".into()],
        layers_to_transform: Some(vec![0, 2]),
        rank_pattern,
        backbone_dtype: BackboneDtype::BF16,
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: AdapterConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.model_type, cfg.model_type);
    assert_eq!(decoded.lora_rank, cfg.lora_rank);
    assert_eq!(decoded.lora_alpha, cfg.lora_alpha);
    assert_eq!(decoded.use_rslora, cfg.use_rslora);
    assert_eq!(decoded.target_modules, cfg.target_modules);
    assert_eq!(decoded.layers_to_transform, cfg.layers_to_transform);
    assert_eq!(decoded.rank_pattern, cfg.rank_pattern);
    assert_eq!(decoded.backbone_dtype, cfg.backbone_dtype);
}

#[test]
fn adapter_config_default_optional_fields() {
    let json = r#"{
        "model_type": "bert",
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "use_rslora": false,
        "target_modules": ["query"]
    }"#;
    let cfg: AdapterConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.layers_to_transform.is_none());
    assert!(cfg.rank_pattern.is_empty());
    assert_eq!(cfg.backbone_dtype, BackboneDtype::F32);
}

#[test]
fn save_load_adapter_roundtrip() {
    let device = cpu();
    let dir = tempfile::tempdir().unwrap();

    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), &device).unwrap();

    let mut tensors = HashMap::new();
    tensors.insert("layer.0.query.lora_a".to_string(), a.clone());
    tensors.insert("layer.0.query.lora_b".to_string(), b.clone());

    let cfg = AdapterConfig {
        model_type: "bert".into(),
        lora_rank: 2,
        lora_alpha: 4.0,
        use_rslora: false,
        target_modules: vec!["query".into()],
        layers_to_transform: None,
        rank_pattern: HashMap::new(),
        backbone_dtype: BackboneDtype::F32,
    };

    save_adapter(dir.path(), &tensors, &cfg).unwrap();
    let (cfg_back, tensors_back): (AdapterConfig, _) = load_adapter(dir.path(), &device).unwrap();

    assert_eq!(cfg_back.lora_rank, cfg.lora_rank);
    assert_eq!(cfg_back.target_modules, cfg.target_modules);
    assert_eq!(tensors_back.len(), 2);

    for key in ["layer.0.query.lora_a", "layer.0.query.lora_b"] {
        let original = tensors.get(key).unwrap().flatten_all().unwrap();
        let loaded = tensors_back.get(key).unwrap().flatten_all().unwrap();
        let orig_vec: Vec<f32> = original.to_vec1().unwrap();
        let load_vec: Vec<f32> = loaded.to_vec1().unwrap();
        assert_eq!(orig_vec, load_vec, "tensor {key} did not round-trip");
    }
}

#[test]
fn lora_build_config_frozen_is_no_op() {
    let cfg = LoraBuildConfig::frozen();
    assert!(cfg.target_modules.is_empty());
    assert!(cfg.layers_to_transform.is_none());
    assert!(cfg.rank_pattern.is_empty());
    assert!(cfg.lora_dropout.is_none());
    // No module should match an empty target list.
    assert!(!should_apply_lora(
        "query",
        cfg.target_modules,
        0,
        cfg.layers_to_transform
    ));
}
