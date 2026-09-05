//! CPU-hermetic proof for `closes_escape: esc-086-bert-loader-legacy-gamma-beta-layernorm-names`.
//!
//! Derives temp checkpoints from `cookbook/fixtures/tiny_bert/model.safetensors`
//! (via `candle_core::safetensors::load`/`save`) rather than shipping a second
//! binary fixture. The fixture's three `LayerNorm` sites (`embeddings.LayerNorm`,
//! `encoder.layer.0.attention.output.LayerNorm`, `encoder.layer.0.output.LayerNorm`,
//! each `weight`/`bias` shape `[32]` F32) carry all-ones/all-zeros affine params, so
//! every tensor is first PERTURBED to distinct, finite, per-site, non-unit/non-zero
//! values before any modern/legacy pair is derived from it -- otherwise a
//! `VarMap`/`Zeros`-style `Init::Const` default could masquerade as a correct load
//! (see arm 0 below, and `crate::layer_norm::LayerNorm::new`'s doc on why the
//! frozen `from_mmaped_safetensors` builder used here has NO such fallback in the
//! first place -- the perturbation only guards this TEST's own oracle, not a real
//! code-path ambiguity).
//!
//! No network, no GPU, no `parity-test`/`golden-parity`/`live-hub-tests` feature.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use jammi_encoders::{Bert, BertConfig, EncoderError, Pooling};
use jammi_lora::LoraBuildConfig;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
}

fn fixture_config() -> BertConfig {
    let config_str = std::fs::read_to_string(fixture_dir().join("config.json"))
        .expect("read tiny_bert config.json");
    serde_json::from_str(&config_str).expect("parse tiny_bert BertConfig")
}

/// Every tensor name in the raw (unrenamed) fixture whose LAST TWO
/// `.`-segments are `LayerNorm.weight`/`LayerNorm.bias` -- the exact
/// suffix-anchored predicate `esc-086`'s `observable` and
/// `ci/scripts/perf/convert_legacy_bert_checkpoint.py` both use.
fn is_layer_norm_weight(name: &str) -> bool {
    name.ends_with(".LayerNorm.weight") || name == "LayerNorm.weight"
}

/// Loads the raw fixture and PERTURBS every `*.LayerNorm.{weight,bias}`
/// tensor to distinct, finite, per-site, non-unit/non-zero F32 values
/// (`weight_i = 0.5 + 0.01*(site*32+i)`, `bias_i = -0.25 + 0.02*(site*32+i)`),
/// leaving every other tensor untouched. Sites are visited in a fixed,
/// sorted-key order so the "distinct across sites" property (arm 0) is
/// deterministic regardless of `HashMap` iteration order.
fn perturbed_fixture_tensors() -> HashMap<String, Tensor> {
    let device = Device::Cpu;
    let mut tensors =
        candle_core::safetensors::load(fixture_dir().join("model.safetensors"), &device)
            .expect("load tiny_bert model.safetensors");

    let mut ln_prefixes: Vec<String> = tensors
        .keys()
        .filter(|k| is_layer_norm_weight(k))
        .map(|k| k.strip_suffix(".weight").unwrap().to_string())
        .collect();
    ln_prefixes.sort();
    assert_eq!(
        ln_prefixes.len(),
        3,
        "tiny_bert fixture must carry exactly 3 LayerNorm sites, got {ln_prefixes:?}"
    );

    for (site, prefix) in ln_prefixes.iter().enumerate() {
        let w_name = format!("{prefix}.weight");
        let b_name = format!("{prefix}.bias");
        let hidden = tensors[&w_name]
            .dims1()
            .expect("LayerNorm weight is rank-1");
        assert_eq!(hidden, 32, "tiny_bert LayerNorm hidden must be 32");

        let w_vals: Vec<f32> = (0..hidden)
            .map(|i| 0.5 + 0.01 * (site * hidden + i) as f32)
            .collect();
        let b_vals: Vec<f32> = (0..hidden)
            .map(|i| -0.25 + 0.02 * (site * hidden + i) as f32)
            .collect();
        tensors.insert(
            w_name,
            Tensor::from_vec(w_vals, (hidden,), &device).unwrap(),
        );
        tensors.insert(
            b_name,
            Tensor::from_vec(b_vals, (hidden,), &device).unwrap(),
        );
    }
    tensors
}

/// Renames every tensor whose name ends `.LayerNorm.weight`/`.LayerNorm.bias`
/// to end `.LayerNorm.gamma`/`.LayerNorm.beta` (byte-identical values),
/// leaving every other tensor untouched -- the exact rename
/// `esc-086`'s `observable` and `ci/scripts/perf/convert_legacy_bert_checkpoint.py`
/// both perform.
fn to_legacy_names(tensors: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    tensors
        .iter()
        .map(|(name, t)| {
            let renamed = if let Some(stem) = name.strip_suffix(".LayerNorm.weight") {
                format!("{stem}.LayerNorm.gamma")
            } else if let Some(stem) = name.strip_suffix(".LayerNorm.bias") {
                format!("{stem}.LayerNorm.beta")
            } else {
                name.clone()
            };
            (renamed, t.clone())
        })
        .collect()
}

fn save_tensors(tensors: &HashMap<String, Tensor>, dir: &Path, filename: &str) -> PathBuf {
    let path = dir.join(filename);
    candle_core::safetensors::save(tensors, &path).expect("write derived safetensors checkpoint");
    path
}

fn build_bert(weights_path: &Path, config: &BertConfig) -> Result<Bert, EncoderError> {
    let varmap = VarMap::new();
    Bert::builder()
        .pooling(Pooling::Mean)
        .lora(LoraBuildConfig::frozen())
        .backbone_dtype(DType::F32)
        .adapter(None)
        .build(&[weights_path], config, &Device::Cpu, &varmap)
}

/// `jammi_encoders::Bert` does not implement `Debug`, so `Result::expect_err`/
/// `unwrap_err` (both bound on `T: Debug`) cannot be used directly on a
/// `Result<Bert, EncoderError>` -- this is the manual equivalent, panicking
/// with `msg` on an unexpected `Ok`.
fn expect_build_err(result: Result<Bert, EncoderError>, msg: &str) -> EncoderError {
    match result {
        Ok(_) => panic!("{msg}: got Ok(Bert), expected Err"),
        Err(e) => e,
    }
}

fn fixed_inputs() -> (Tensor, Tensor) {
    let device = Device::Cpu;
    let input_ids = Tensor::new(&[[2u32, 121, 124, 1, 3], [2, 121, 1, 3, 0]], &device).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 1, 1, 1], [1, 1, 1, 1, 0]], &device).unwrap();
    (input_ids, mask)
}

/// Arm 0 (non-vacuity): every perturbed LayerNorm tensor is finite, not
/// all-ones/all-zeros, and distinct across sites -- otherwise arm 2 could
/// not tell a correct alias-load from an `Init::Const` default (which this
/// frozen builder does not even have, but the ones/zeros starting values
/// the fixture ships with would make a "did it load the RIGHT tensor at
/// all" bug invisible).
#[test]
fn arm0_perturbed_fixture_is_finite_and_non_trivial_and_distinct_across_sites() {
    let tensors = perturbed_fixture_tensors();
    let mut ln_prefixes: Vec<String> = tensors
        .keys()
        .filter(|k| is_layer_norm_weight(k))
        .map(|k| k.strip_suffix(".weight").unwrap().to_string())
        .collect();
    ln_prefixes.sort();

    let mut all_weight_vecs = Vec::new();
    let mut all_bias_vecs = Vec::new();
    for prefix in &ln_prefixes {
        let w: Vec<f32> = tensors[&format!("{prefix}.weight")]
            .to_vec1()
            .expect("weight to_vec1");
        let b: Vec<f32> = tensors[&format!("{prefix}.bias")]
            .to_vec1()
            .expect("bias to_vec1");
        assert!(
            w.iter().all(|v| v.is_finite()),
            "{prefix}.weight must be all-finite"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "{prefix}.bias must be all-finite"
        );
        assert!(
            !w.iter().all(|&v| v == 1.0),
            "{prefix}.weight must not still be all-ones"
        );
        assert!(
            !b.iter().all(|&v| v == 0.0),
            "{prefix}.bias must not still be all-zeros"
        );
        all_weight_vecs.push(w);
        all_bias_vecs.push(b);
    }
    for i in 0..all_weight_vecs.len() {
        for j in (i + 1)..all_weight_vecs.len() {
            assert_ne!(
                all_weight_vecs[i], all_weight_vecs[j],
                "sites {i} and {j} must carry distinct weight values"
            );
            assert_ne!(
                all_bias_vecs[i], all_bias_vecs[j],
                "sites {i} and {j} must carry distinct bias values"
            );
        }
    }
}

/// Arm 1 (alias): building from the LEGACY-named perturbed checkpoint
/// succeeds.
#[test]
fn arm1_build_succeeds_from_legacy_named_checkpoint() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let perturbed = perturbed_fixture_tensors();
    let legacy = to_legacy_names(&perturbed);
    let legacy_path = save_tensors(&legacy, tmp.path(), "legacy.safetensors");

    build_bert(&legacy_path, &config).expect("legacy-named checkpoint must build Ok");
}

/// Arm 2 (silent-wrong): a fixed forward through the legacy-built and the
/// modern-built (both from the SAME perturbed values) models must be
/// exactly, bitwise identical -- an `is_finite()` scan first, over EVERY
/// element of BOTH outputs, then exact bitwise equality via `to_bits`.
#[test]
fn arm2_legacy_and_modern_forward_are_bitwise_identical() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let perturbed = perturbed_fixture_tensors();
    let legacy = to_legacy_names(&perturbed);

    let modern_path = save_tensors(&perturbed, tmp.path(), "modern.safetensors");
    let legacy_path = save_tensors(&legacy, tmp.path(), "legacy.safetensors");

    let modern_bert = build_bert(&modern_path, &config).expect("modern build Ok");
    let legacy_bert = build_bert(&legacy_path, &config).expect("legacy build Ok");

    let (input_ids, mask) = fixed_inputs();
    let modern_out = modern_bert
        .forward_hidden(&input_ids, &mask)
        .expect("modern forward");
    let legacy_out = legacy_bert
        .forward_hidden(&input_ids, &mask)
        .expect("legacy forward");

    assert_eq!(modern_out.dims(), legacy_out.dims());

    let modern_v: Vec<f32> = modern_out.flatten_all().unwrap().to_vec1().unwrap();
    let legacy_v: Vec<f32> = legacy_out.flatten_all().unwrap().to_vec1().unwrap();

    assert!(
        modern_v.iter().all(|v| v.is_finite()),
        "modern output must be all-finite before any bit compare"
    );
    assert!(
        legacy_v.iter().all(|v| v.is_finite()),
        "legacy output must be all-finite before any bit compare"
    );
    assert_eq!(
        modern_v.len(),
        legacy_v.len(),
        "modern/legacy output element counts must match"
    );
    for (i, (m, l)) in modern_v.iter().zip(legacy_v.iter()).enumerate() {
        assert_eq!(
            m.to_bits(),
            l.to_bits(),
            "element {i}: modern {m} (bits {:#x}) vs legacy {l} (bits {:#x}) -- \
             must be EXACTLY bitwise identical, not merely close",
            m.to_bits(),
            l.to_bits()
        );
    }
}

/// Arm 4 (collision): a checkpoint carrying BOTH
/// `embeddings.LayerNorm.gamma` AND `embeddings.LayerNorm.weight`
/// (different values) must be `Err`, naming the prefix and both names.
#[test]
fn arm4_weight_gamma_collision_is_refused_and_named() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let mut tensors = perturbed_fixture_tensors();

    // Insert a DIFFERENT-valued `gamma` alongside the existing `weight` at
    // the embeddings LayerNorm.
    let existing_w: Vec<f32> = tensors["embeddings.LayerNorm.weight"].to_vec1().unwrap();
    let colliding: Vec<f32> = existing_w.iter().map(|v| v + 100.0).collect();
    tensors.insert(
        "embeddings.LayerNorm.gamma".to_string(),
        Tensor::from_vec(colliding, (existing_w.len(),), &Device::Cpu).unwrap(),
    );

    let path = save_tensors(&tensors, tmp.path(), "collision.safetensors");
    let err = expect_build_err(
        build_bert(&path, &config),
        "weight+gamma collision must be Err",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("embeddings.LayerNorm"),
        "message must name the prefix: {msg}"
    );
    assert!(
        msg.contains("weight") && msg.contains("gamma"),
        "message must name both candidate tensor names: {msg}"
    );
}

/// The beta+bias double-collision variant: the weight axis is still what
/// gets reported (deterministic), never the bias axis.
#[test]
fn arm4b_double_collision_reports_weight_axis() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let mut tensors = perturbed_fixture_tensors();

    let existing_w: Vec<f32> = tensors["embeddings.LayerNorm.weight"].to_vec1().unwrap();
    let existing_b: Vec<f32> = tensors["embeddings.LayerNorm.bias"].to_vec1().unwrap();
    tensors.insert(
        "embeddings.LayerNorm.gamma".to_string(),
        Tensor::from_vec(
            existing_w.iter().map(|v| v + 100.0).collect::<Vec<_>>(),
            (existing_w.len(),),
            &Device::Cpu,
        )
        .unwrap(),
    );
    tensors.insert(
        "embeddings.LayerNorm.beta".to_string(),
        Tensor::from_vec(
            existing_b.iter().map(|v| v + 100.0).collect::<Vec<_>>(),
            (existing_b.len(),),
            &Device::Cpu,
        )
        .unwrap(),
    );

    let path = save_tensors(&tensors, tmp.path(), "double_collision.safetensors");
    let err = expect_build_err(build_bert(&path, &config), "double collision must be Err");
    let msg = err.to_string();
    assert!(
        msg.contains("weight") && msg.contains("gamma"),
        "double collision must report the WEIGHT axis: {msg}"
    );
}

/// Arm 5 (legacy-then-missing): `gamma` present, neither `bias` nor `beta`
/// present at all -- `Err` naming the prefix and BOTH `bias` and `beta`,
/// never a bare `cannot find tensor`.
#[test]
fn arm5_legacy_weight_with_no_bias_or_beta_is_refused_and_names_both() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let perturbed = perturbed_fixture_tensors();
    let mut legacy = to_legacy_names(&perturbed);
    // Remove BOTH the (already-renamed) `beta` and any `bias` at the
    // embeddings LayerNorm, leaving `gamma` as the only affine tensor there.
    legacy.remove("embeddings.LayerNorm.beta");
    legacy.remove("embeddings.LayerNorm.bias");
    assert!(legacy.contains_key("embeddings.LayerNorm.gamma"));

    let path = save_tensors(&legacy, tmp.path(), "legacy_then_missing.safetensors");
    let err = expect_build_err(
        build_bert(&path, &config),
        "missing bias AND beta must be Err",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("embeddings.LayerNorm"),
        "message must name the prefix: {msg}"
    );
    assert!(
        msg.contains("bias") && msg.contains("beta"),
        "message must name BOTH `bias` and `beta`, never a bare \
         `cannot find tensor`: {msg}"
    );
    assert!(
        !msg.trim_start().starts_with("cannot find tensor"),
        "message must not be a bare `cannot find tensor`: {msg}"
    );
}

/// Arm 6a (boundary, non-vacuous -- `#423` narrow-fix round 2 / B1a): a
/// parent-level `embeddings.gamma` (one segment ABOVE `embeddings.LayerNorm`,
/// no `LayerNorm` segment in ITS OWN name at all) must NOT be aliased into
/// `embeddings.LayerNorm`'s weight axis, even though `embeddings.LayerNorm`
/// itself is genuinely `LayerNorm`-keyed and its `bias` axis is genuinely
/// present and readable.
///
/// `embeddings.LayerNorm.bias` is deliberately KEPT here (only `.weight` is
/// removed): the ORIGINAL version of this arm deleted BOTH axes, which made
/// `build` return `Err` under EVERY implementation -- including a
/// PARENT-PROBING one that (incorrectly) falls back to a tensor one level up
/// when the exact prefix has neither `weight` nor `gamma` -- so it could
/// never actually distinguish correct code from that bug. Keeping `bias`
/// present means a parent-probing implementation would have succeeded
/// (`build` returns `Ok`) here, and this arm now catches that.
///
/// Proved by hand once (not shipped, restored after observing the failure):
/// temporarily made `LayerNorm::new`'s weight-axis load fall back to
/// `vb.root().set_prefix(<prefix's parent>).get_with_hints(hidden_size,
/// "gamma", ..)` whenever `gamma` is absent at the exact prefix but present
/// one level up (`VarBuilder::root`/`set_prefix` make this directly
/// expressible with no signature change, since a `VarBuilder` carries a
/// `data: Arc<..>` shared across `root()`/`pp()`/`set_prefix()` calls --
/// candle-nn-0.11.0's `var_builder.rs:129-146`). With that mutation in
/// place, this test turned RED:
/// `arm6a_parent_level_gamma_is_not_aliased: a parent-level \`embeddings.gamma\`
/// must not be aliased into \`embeddings.LayerNorm\`: got Ok(Bert), expected Err`
/// -- confirming the arm bites a parent-probing implementation, not just an
/// always-Err one.
#[test]
fn arm6a_parent_level_gamma_is_not_aliased() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let mut tensors = perturbed_fixture_tensors();

    let w = tensors.remove("embeddings.LayerNorm.weight").unwrap();
    assert!(
        tensors.contains_key("embeddings.LayerNorm.bias"),
        "the bias axis must stay present -- deleting it too would make this arm vacuous again \
         (every implementation errors when BOTH axes are missing, correct or not)"
    );
    // Re-home the removed weight one level UP, as a parent-level `gamma`.
    tensors.insert("embeddings.gamma".to_string(), w);

    let path = save_tensors(&tensors, tmp.path(), "parent_level_gamma.safetensors");
    let err = expect_build_err(
        build_bert(&path, &config),
        "a parent-level `embeddings.gamma` must not be aliased into `embeddings.LayerNorm`",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("embeddings.LayerNorm"),
        "message must name the prefix: {msg}"
    );
    assert!(
        msg.contains("weight") && msg.contains("gamma"),
        "message must name the weight axis's own candidates -- \
         `embeddings.LayerNorm.gamma` is genuinely absent (the only `gamma` tensor in this \
         checkpoint sits one level UP, at `embeddings.gamma`, which a correct alias must never \
         consult): {msg}"
    );
}

// Arm 6b (boundary) moved into `crate::layer_norm`'s own `#[cfg(test)]`
// module as a DIRECT seam test (B1b): no in-tree model ever builds a
// `VarBuilder` at a synthetic `embeddings.LayerNormX`-style prefix, so that
// boundary is now exercised against a REAL non-`LayerNorm`-keyed production
// prefix (DistilBERT's `sa_layer_norm`) instead -- see
// `layer_norm::tests::direct_seam_non_layer_norm_keyed_prefix_containing_layer_norm_substring_is_not_aliased`.

/// Arm 7 (MIXED, B3): each of the fixture's three `LayerNorm` sites uses a
/// DIFFERENT per-site/per-axis naming -- `embeddings.LayerNorm` is FULLY
/// legacy (`gamma`+`beta`), `encoder.layer.0.attention.output.LayerNorm` is
/// left FULLY modern (untouched), and `encoder.layer.0.output.LayerNorm` is
/// PER-AXIS mixed (`gamma` for the weight axis, the modern `bias` for the
/// bias axis). `resolve_affine_names` resolves each axis of each site
/// INDEPENDENTLY (see its own doc's presence lattice) -- this is the
/// realistic shape a half-converted or partially-migrated checkpoint takes,
/// not every site (or even every axis within one site) necessarily renamed
/// together. Values are untouched by the rename, so a correct build's
/// forward must be bitwise identical to the all-modern control.
#[test]
fn arm7_mixed_per_site_and_per_axis_naming_builds_and_matches_all_modern() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let perturbed = perturbed_fixture_tensors();

    let mut ln_prefixes: Vec<String> = perturbed
        .keys()
        .filter(|k| is_layer_norm_weight(k))
        .map(|k| k.strip_suffix(".weight").unwrap().to_string())
        .collect();
    ln_prefixes.sort();
    assert_eq!(
        ln_prefixes.len(),
        3,
        "expected exactly 3 LayerNorm sites, got {ln_prefixes:?}"
    );
    let site1 = &ln_prefixes[0]; // "embeddings.LayerNorm" -- fully legacy.
    let site3 = &ln_prefixes[2]; // "encoder.layer.0.output.LayerNorm" -- mixed.
                                 // site2 ("encoder.layer.0.attention.output.LayerNorm") is left untouched.

    let mut mixed = perturbed.clone();

    let w1 = mixed.remove(&format!("{site1}.weight")).unwrap();
    let b1 = mixed.remove(&format!("{site1}.bias")).unwrap();
    mixed.insert(format!("{site1}.gamma"), w1);
    mixed.insert(format!("{site1}.beta"), b1);

    let w3 = mixed.remove(&format!("{site3}.weight")).unwrap();
    mixed.insert(format!("{site3}.gamma"), w3);
    // `{site3}.bias` is left in place -- the modern name, on purpose: this
    // site's TWO axes are named inconsistently with each other.

    let modern_path = save_tensors(&perturbed, tmp.path(), "modern.safetensors");
    let mixed_path = save_tensors(&mixed, tmp.path(), "mixed.safetensors");

    let modern_bert = build_bert(&modern_path, &config).expect("modern build Ok");
    let mixed_bert =
        build_bert(&mixed_path, &config).expect("mixed per-site/per-axis naming build must be Ok");

    let (input_ids, mask) = fixed_inputs();
    let modern_out = modern_bert
        .forward_hidden(&input_ids, &mask)
        .expect("modern forward");
    let mixed_out = mixed_bert
        .forward_hidden(&input_ids, &mask)
        .expect("mixed forward");

    assert_eq!(modern_out.dims(), mixed_out.dims());
    let modern_v: Vec<f32> = modern_out.flatten_all().unwrap().to_vec1().unwrap();
    let mixed_v: Vec<f32> = mixed_out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        modern_v.iter().all(|v| v.is_finite()),
        "modern output must be all-finite before any bit compare"
    );
    assert!(
        mixed_v.iter().all(|v| v.is_finite()),
        "mixed output must be all-finite before any bit compare"
    );
    assert_eq!(modern_v.len(), mixed_v.len());
    for (i, (m, x)) in modern_v.iter().zip(mixed_v.iter()).enumerate() {
        assert_eq!(
            m.to_bits(),
            x.to_bits(),
            "element {i}: modern {m} (bits {:#x}) vs mixed {x} (bits {:#x}) -- must be \
             EXACTLY bitwise identical, not merely close",
            m.to_bits(),
            x.to_bits()
        );
    }
}

/// Arm 8 (coverage gap (i)): the layout the real stock `bert-base-uncased`
/// checkpoint actually ships -- every tensor re-keyed under a `bert.`
/// prefix (`bert.rs:526-535` probes `bert.embeddings.word_embeddings.weight`
/// and selects that prefix over the root-level layout). Building from a
/// LEGACY-named checkpoint wrapped this way must be `Ok`, and its forward
/// must be bitwise identical to the root-level (unwrapped) legacy build --
/// the `bert.` wrapper is purely a naming layer, orthogonal to the
/// legacy-name alias this whole file proves.
#[test]
fn arm8_bert_wrapped_legacy_layout_matches_root_level_legacy_build() {
    let config = fixture_config();
    let tmp = tempfile::tempdir().unwrap();
    let perturbed = perturbed_fixture_tensors();
    let legacy = to_legacy_names(&perturbed);

    let root_legacy_path = save_tensors(&legacy, tmp.path(), "root_legacy.safetensors");
    let wrapped_legacy: HashMap<String, Tensor> = legacy
        .iter()
        .map(|(name, t)| (format!("bert.{name}"), t.clone()))
        .collect();
    let wrapped_path = save_tensors(
        &wrapped_legacy,
        tmp.path(),
        "bert_wrapped_legacy.safetensors",
    );

    let root_bert = build_bert(&root_legacy_path, &config).expect("root-level legacy build Ok");
    let wrapped_bert =
        build_bert(&wrapped_path, &config).expect("bert.-wrapped legacy build must be Ok");

    let (input_ids, mask) = fixed_inputs();
    let root_out = root_bert
        .forward_hidden(&input_ids, &mask)
        .expect("root forward");
    let wrapped_out = wrapped_bert
        .forward_hidden(&input_ids, &mask)
        .expect("wrapped forward");

    assert_eq!(root_out.dims(), wrapped_out.dims());
    let root_v: Vec<f32> = root_out.flatten_all().unwrap().to_vec1().unwrap();
    let wrapped_v: Vec<f32> = wrapped_out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        root_v.iter().all(|v| v.is_finite()),
        "root output must be all-finite before any bit compare"
    );
    assert!(
        wrapped_v.iter().all(|v| v.is_finite()),
        "wrapped output must be all-finite before any bit compare"
    );
    assert_eq!(root_v.len(), wrapped_v.len());
    for (i, (r, w)) in root_v.iter().zip(wrapped_v.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            w.to_bits(),
            "element {i}: root {r} (bits {:#x}) vs bert.-wrapped {w} (bits {:#x}) -- must be \
             EXACTLY bitwise identical, not merely close",
            r.to_bits(),
            w.to_bits()
        );
    }
}
