//! Integration tests for the `jammi-lora` public surface. CPU-only, hermetic.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{
    effective_rank, load_adapter, save_adapter, should_apply_lora, AdapterConfig, ComputePrecision,
    FrozenBase, LoraBuildConfig, LoraError, LoraInitMode, LoraLinear, MaybeLoraLinear, Tower,
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
    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::ZerosB,
        None,
        0,
        &varmap,
        &vb,
    )
    .unwrap();
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
    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        0,
        &varmap,
        &vb,
    )
    .unwrap();
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
    let lora = LoraLinear::new_simple(base, 4, 8.0, 0, &varmap, &vb).unwrap();
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
        false,
    )
    .unwrap();
    // For RSLoRA we have to construct via `new` to exercise the use_rslora
    // path; we then overwrite A/B with ones so the delta is observable.
    let mut rslora = LoraLinear::new(
        base,
        4,
        4.0,
        true,
        LoraInitMode::ZerosB,
        None,
        0,
        &varmap,
        &vb,
    )
    .unwrap();
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

/// Table-driven: `new` and `from_loaded` must compute BIT-EQUAL `f64`
/// scaling for the same `(alpha, rank, use_rslora)` triple — both route
/// through the crate's single `lora_scaling` primitive (see its doc), so a
/// disagreement here would mean a reload of a saved adapter silently applies
/// a different scaling than the run that trained it (esc-041).
#[test]
fn new_and_from_loaded_pin_bit_equal_scaling() {
    let device = cpu();
    let cases: &[(f64, usize, bool)] = &[
        (8.0, 4, false),
        (8.0, 4, true),
        (16.0, 8, false),
        (16.0, 8, true),
        (1.0, 1, false),
        (1.0, 1, true),
        (32.0, 16, false),
        (32.0, 16, true),
    ];
    for &(alpha, rank, use_rslora) in cases {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let base = build_base(8, 16, &device);
        let new_layer = LoraLinear::new(
            base,
            rank,
            alpha,
            use_rslora,
            LoraInitMode::ZerosB,
            None,
            0,
            &varmap,
            &vb,
        )
        .unwrap();

        let lora_a = Tensor::zeros((rank, 8), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((16, rank), DType::F32, &device).unwrap();
        let loaded_layer = LoraLinear::from_loaded(
            build_base(8, 16, &device),
            lora_a,
            lora_b,
            alpha,
            use_rslora,
        )
        .unwrap();

        let expected = if use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        };

        assert_eq!(
            new_layer.scaling().to_bits(),
            expected.to_bits(),
            "new(): alpha={alpha} rank={rank} use_rslora={use_rslora}"
        );
        assert_eq!(
            loaded_layer.scaling().to_bits(),
            expected.to_bits(),
            "from_loaded(): alpha={alpha} rank={rank} use_rslora={use_rslora}"
        );
        assert_eq!(
            new_layer.scaling().to_bits(),
            loaded_layer.scaling().to_bits(),
            "new() and from_loaded() disagree: alpha={alpha} rank={rank} use_rslora={use_rslora}"
        );
    }
}

/// Two LoRA sites under the SAME `VarBuilder` prefix, into the same
/// `VarMap`, is a typed refusal — not a silent alias.
///
/// Candle's `VarMap::get` returns the ALREADY-REGISTERED `Var` for a name it
/// has seen, so without this gate the second construction hands back the
/// FIRST site's `Var`s: half the intended trainable parameters, one gradient
/// stream feeding two sites, and an exported adapter whose two entries are
/// byte-identical. Every one of those is a confidently wrong number, never an
/// error — exactly the class this refusal exists for (it is what the two
/// OpenCLIP towers did to each other before they were given disjoint adapter
/// key roots).
#[test]
fn a_second_site_on_the_same_varmap_key_is_a_typed_refusal() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let first = LoraLinear::new(
        build_base(8, 16, &device),
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        7,
        &varmap,
        &vb.pp("site"),
    );
    assert!(
        first.is_ok(),
        "the FIRST site under a fresh prefix must construct"
    );

    let second = LoraLinear::new(
        build_base(8, 16, &device),
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        7,
        &varmap,
        &vb.pp("site"),
    );
    // `LoraLinear` intentionally doesn't derive `Debug`; match instead.
    let err = match second {
        Ok(_) => panic!("a second site on `site.lora_a`/`site.lora_b` must be refused"),
        Err(e) => e,
    };
    assert!(
        matches!(err, LoraError::Config(_)),
        "the collision must be a typed Config refusal, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("site.lora_a") && msg.contains("site.lora_b"),
        "the refusal must name BOTH colliding keys so the caller can find them, got: {msg}"
    );

    // Negative control on the same `VarMap`: a DISTINCT prefix is fine, so
    // the refusal above is about the key collision and not about the varmap
    // having been written to at all.
    assert!(
        LoraLinear::new(
            build_base(8, 16, &device),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            7,
            &varmap,
            &vb.pp("other_site"),
        )
        .is_ok(),
        "a distinct prefix in the same VarMap must still construct"
    );
}

/// The INFERENCE path is untouched by the collision gate: an adapter-file
/// `VarBuilder` never registers anything in `varmap`, so building two towers
/// from the same saved adapter into the same (unused) `VarMap` still works.
///
/// Without this control the gate above could be over-broad — refusing the
/// legitimate "serve two copies" case — and the positive test alone would not
/// notice.
#[test]
fn loading_the_same_adapter_twice_into_one_varmap_is_not_a_collision() {
    let device = cpu();
    let dir = tempfile::tempdir().unwrap();

    // Train one site, save it.
    let train_map = VarMap::new();
    let train_vb = VarBuilder::from_varmap(&train_map, DType::F32, &device);
    let trained = LoraLinear::new(
        build_base(8, 16, &device),
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        11,
        &train_map,
        &train_vb.pp("site"),
    )
    .unwrap();
    let wrapped = MaybeLoraLinear::Lora(trained);
    let weights = wrapped.named_weights("site").unwrap();
    let targets: Vec<String> = vec!["site".into()];
    let layers = None;
    let rank_pattern = HashMap::new();
    let build_cfg = LoraBuildConfig {
        target_modules: &targets,
        layers_to_transform: &layers,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &rank_pattern,
        init_mode: LoraInitMode::Gaussian,
        seed: 11,
    };
    save_adapter(
        dir.path(),
        &weights,
        &AdapterConfig::from_build("bert", &build_cfg, ComputePrecision::F32),
    )
    .unwrap();

    // Two constructions from that file, sharing ONE VarMap.
    let serve_map = VarMap::new();
    let file = dir.path().join("adapter.safetensors");
    for attempt in 0..2 {
        let serve_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[file.as_path()], DType::F32, &device).unwrap()
        };
        let built = LoraLinear::new(
            build_base(8, 16, &device),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            11,
            &serve_map,
            &serve_vb.pp("site"),
        );
        assert!(
            built.is_ok(),
            "inference construction #{attempt} from an adapter file must not trip the \
             training-path collision gate"
        );
    }
    assert!(
        serve_map.data().lock().unwrap().is_empty(),
        "the inference path must not register anything in the VarMap — if it did, the \
         control above would be passing for the wrong reason"
    );
}

/// Domain boundary (family D / K2): both constructors must typed-refuse
/// `rank == 0` rather than let it reach `alpha / 0`.
#[test]
fn rank_zero_is_a_typed_refusal_in_new_and_from_loaded() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let base = build_base(8, 16, &device);

    let new_result = LoraLinear::new(
        base,
        0,
        8.0,
        false,
        LoraInitMode::ZerosB,
        None,
        0,
        &varmap,
        &vb,
    );
    // `LoraLinear` intentionally doesn't derive `Debug` (candle's `Linear`
    // doesn't either) so `unwrap_err()` isn't available; match instead.
    let new_err = match new_result {
        Ok(_) => panic!("new(rank=0) must be refused, not constructed"),
        Err(e) => e,
    };
    assert!(
        matches!(new_err, LoraError::Config(_)),
        "new(rank=0) must be a typed Config refusal, got {new_err:?}"
    );

    let lora_a = Tensor::zeros((0, 8), DType::F32, &device).unwrap();
    let lora_b = Tensor::zeros((16, 0), DType::F32, &device).unwrap();
    let loaded_result =
        LoraLinear::from_loaded(build_base(8, 16, &device), lora_a, lora_b, 8.0, false);
    let loaded_err = match loaded_result {
        Ok(_) => panic!("from_loaded(rank=0) must be refused, not constructed"),
        Err(e) => e,
    };
    assert!(
        matches!(loaded_err, LoraError::Config(_)),
        "from_loaded(rank=0) must be a typed Config refusal, got {loaded_err:?}"
    );
}

#[test]
fn maybe_lora_linear_frozen_forward_matches_underlying() {
    let device = cpu();
    let base = build_base(8, 16, &device);
    let frozen = MaybeLoraLinear::Frozen(FrozenBase::Dense(build_base(8, 16, &device)));

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

    let frozen = MaybeLoraLinear::Frozen(FrozenBase::Dense(build_base(8, 16, &device)));
    assert!(frozen.named_weights("query").unwrap().is_empty());

    let lora = MaybeLoraLinear::Lora(
        LoraLinear::new_simple(build_base(8, 16, &device), 4, 8.0, 0, &varmap, &vb).unwrap(),
    );
    let weights = lora.named_weights("query").unwrap();
    assert_eq!(weights.len(), 2);
    assert!(weights.contains_key("query.lora_a"));
    assert!(weights.contains_key("query.lora_b"));
}

#[test]
fn should_apply_lora_exact_match() {
    let targets = vec!["query".to_string()];
    assert!(should_apply_lora("query", &targets, Some(0), &None));
    assert!(should_apply_lora(
        "attention.self.query",
        &targets,
        Some(0),
        &None
    ));
    assert!(!should_apply_lora("key", &targets, Some(0), &None));
    assert!(!should_apply_lora("queryless", &targets, Some(0), &None));
}

#[test]
fn should_apply_lora_all_linear() {
    let targets = vec!["all-linear".to_string()];
    assert!(should_apply_lora("query", &targets, Some(0), &None));
    assert!(should_apply_lora(
        "anything.at.all",
        &targets,
        Some(7),
        &None
    ));
}

#[test]
fn should_apply_lora_layer_filter() {
    let targets = vec!["all-linear".to_string()];
    let layers = Some(vec![2usize, 5]);
    assert!(should_apply_lora("query", &targets, Some(2), &layers));
    assert!(should_apply_lora("query", &targets, Some(5), &layers));
    assert!(!should_apply_lora("query", &targets, Some(0), &layers));
    assert!(!should_apply_lora("query", &targets, Some(3), &layers));
}

/// The UNINDEXED site (`layer_idx == None`) — a linear that belongs to no
/// numbered block at all, e.g. a CLAP audio-projection `linear1`. PEFT's own
/// `check_target_module_exists` (`peft/src/peft/tuners/tuners_utils.py:
/// 2353-2389`) sets `layer_index = None` for a key with no numbered segment
/// and then `target_module_found = False` whenever `layers_to_transform` is
/// set, so:
///
/// - with a layer filter active, an unindexed site is NEVER adapted —
///   including under `all-linear`, which is a NAME wildcard, not a layer
///   one (the negative control that a naive "no index, so nothing to
///   reject" implementation would silently fail);
/// - with no filter, the ordinary selector match decides, exactly as for an
///   indexed site.
#[test]
fn should_apply_lora_unindexed_site_obeys_the_layer_filter() {
    let all_linear = vec!["all-linear".to_string()];
    let by_name = vec!["linear1".to_string()];
    let layers = Some(vec![0usize]);

    // Filter active + no index => never applied, on either selector form.
    assert!(!should_apply_lora("linear1", &all_linear, None, &layers));
    assert!(!should_apply_lora("linear1", &by_name, None, &layers));
    // Same site, same selectors, WITH an index inside the filter => applied
    // (proving the refusal above is the missing index, not the selector).
    assert!(should_apply_lora("linear1", &all_linear, Some(0), &layers));
    assert!(should_apply_lora("linear1", &by_name, Some(0), &layers));

    // No filter => the selector alone decides, index or not.
    assert!(should_apply_lora("linear1", &all_linear, None, &None));
    assert!(should_apply_lora("linear1", &by_name, None, &None));
    assert!(!should_apply_lora("linear2", &by_name, None, &None));
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
        backbone_dtype: ComputePrecision::BF16,
        tower: Some(Tower::Vision),
    };

    let json = serde_json::to_string(&cfg).unwrap();
    let decoded: AdapterConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.tower, cfg.tower);
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
    assert_eq!(cfg.backbone_dtype, ComputePrecision::F32);
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
        backbone_dtype: ComputePrecision::F32,
        tower: None,
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
        Some(0),
        cfg.layers_to_transform
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Backbone-precision parity between the Frozen and LoRA arms
// ─────────────────────────────────────────────────────────────────────────────

/// A reduced backbone dtype must mean the same thing at every linear, whether
/// or not that linear happens to be a LoRA target.
///
/// With `lora_b` initialised to zeros the LoRA contribution is exactly
/// `B @ A @ x = 0`, so `Lora(W)` and `Frozen(W)` are the same function and must
/// agree bit-for-bit. They did not: the `Frozen` arm cast the input down to the
/// weight dtype and ran the matmul at the backbone precision, while the LoRA arm
/// materialised a fresh F32 copy of the frozen weight on every forward and ran
/// the base matmul in F32. A LoRA run therefore silently ignored
/// `backbone_dtype` on precisely the linears it targeted.
///
/// F16 rather than BF16 because candle's CPU matmul accepts `F16 | F32 | F64`
/// only; BF16 on CPU is rejected by the fine-tune config gate instead, and is
/// covered by `bf16_backbone_is_refused_on_cpu`.
mod backbone_precision_parity {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, VarMap};
    use jammi_lora::{FrozenBase, LoraInitMode, LoraLinear, MaybeLoraLinear};

    const IN: usize = 24;
    const OUT: usize = 16;
    const BATCH: usize = 4;

    fn weight(device: &Device) -> Tensor {
        // Deterministic, non-degenerate, and large enough in magnitude that an
        // F32-vs-F16 matmul difference is representable rather than rounding to
        // the same value.
        let data: Vec<f32> = (0..OUT * IN)
            .map(|i| ((i % 17) as f32 - 8.0) / 6.0)
            .collect();
        Tensor::from_vec(data, (OUT, IN), device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    fn input(device: &Device) -> Tensor {
        let data: Vec<f32> = (0..BATCH * IN)
            .map(|i| ((i % 13) as f32 - 6.0) / 5.0)
            .collect();
        Tensor::from_vec(data, (BATCH, IN), device).unwrap()
    }

    fn finite_count(t: &Tensor) -> usize {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .filter(|v| v.is_finite())
            .count()
    }

    fn spread(t: &Tensor) -> f32 {
        let v = t
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        v.iter().fold(f32::MIN, |a, b| a.max(*b)) - v.iter().fold(f32::MAX, |a, b| a.min(*b))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Build both arms over the *same* weight tensor at `dtype`.
    fn arms(device: &Device, dtype: DType) -> (MaybeLoraLinear, MaybeLoraLinear, VarMap) {
        let w = weight(device).to_dtype(dtype).unwrap();
        let frozen = MaybeLoraLinear::Frozen(FrozenBase::Dense(Linear::new(w.clone(), None)));
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
        let lora = LoraLinear::new(
            Linear::new(w, None),
            4,
            8.0,
            false,
            LoraInitMode::ZerosB,
            None,
            42,
            &varmap,
            &vb.pp("site"),
        )
        .unwrap();
        (frozen, MaybeLoraLinear::Lora(lora), varmap)
    }

    #[test]
    fn lora_arm_honours_the_backbone_dtype_like_the_frozen_arm() {
        let device = Device::Cpu;
        let x = input(&device);
        let (frozen, lora, _vm) = arms(&device, DType::F16);

        // The zero-delta premise: without this the comparison below would be
        // measuring a real LoRA contribution, not a precision difference.
        let MaybeLoraLinear::Lora(inner) = &lora else {
            unreachable!("constructed as Lora")
        };
        let b_sum = inner
            .lora_b
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(b_sum, 0.0, "ZerosB must leave lora_b exactly zero");

        let fo = frozen.forward(&x).unwrap();
        let lo = lora.forward(&x).unwrap();

        // Non-vacuity: NaN fails every comparison bound in both directions, and
        // a zeroed matmul would make any two outputs agree.
        assert_eq!(
            finite_count(&fo),
            fo.elem_count(),
            "frozen output non-finite"
        );
        assert_eq!(finite_count(&lo), lo.elem_count(), "lora output non-finite");
        assert!(spread(&fo) > 0.0, "frozen output is constant");
        assert!(spread(&lo) > 0.0, "lora output is constant");

        assert_eq!(
            fo.dtype(),
            lo.dtype(),
            "the two arms must agree on the output dtype"
        );
        let delta = max_abs_diff(&fo, &lo);
        assert_eq!(
            delta, 0.0,
            "with lora_b == 0 the LoRA arm is the frozen arm, but they differ by {delta} \
             at F16 — the LoRA arm is not running the base matmul at the backbone dtype"
        );
    }

    /// Positive control: the harness must discriminate dtype rather than report
    /// agreement everywhere. At F32 both arms have always agreed, and must
    /// continue to.
    #[test]
    fn the_two_arms_agree_at_f32_too() {
        let device = Device::Cpu;
        let x = input(&device);
        let (frozen, lora, _vm) = arms(&device, DType::F32);
        let fo = frozen.forward(&x).unwrap();
        let lo = lora.forward(&x).unwrap();
        assert_eq!(finite_count(&fo), fo.elem_count());
        assert!(spread(&fo) > 0.0);
        assert_eq!(max_abs_diff(&fo, &lo), 0.0);
    }
}
