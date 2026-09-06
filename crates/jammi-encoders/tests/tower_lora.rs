//! LoRA trainability of the three cross-modal towers, on the committed tiny
//! fixtures.
//!
//! Four oracles per tower:
//!
//! * **A1 gradient reachability** — with `all-linear` selected, EVERY
//!   trainable `Var` the tower reports gets a `Some`, finite, non-zero
//!   gradient from one forward+backward, and the site COUNT is the one the
//!   config predicts. This is a mechanism assertion, not a loss-decrease
//!   one (esc-037): a loss that happens to go down proves nothing about
//!   which parameters were reachable.
//! * **A2 eval bit-identity** — `builder().lora(frozen())` equals `load()`
//!   bit-for-bit, and a `ZerosB` adapter installed on every site equals the
//!   unadapted tower bit-for-bit.
//! * **A5 F16** — A1 again with an F16 backbone, so the dtype-following
//!   masks are exercised at a precision where an `f32::MIN` sentinel would
//!   have become `-inf`.
//! * **Adapter round trip** — train, export, save, rebuild through
//!   `.adapter(..)`, and the rebuilt tower's eval output is bit-equal to the
//!   trained one's.
//!
//! # Why `LoraInitMode::Gaussian` and not the default `ZerosB`
//!
//! Under `ZerosB` the B matrix starts at exactly zero, and `dL/dA = B^T ·
//! (dL/dy) · x^T` is therefore exactly `0` at the FIRST backward
//! (`jammi-lora/src/lora_linear.rs`'s `ZerosB` arm). A reachability oracle
//! run on a `ZerosB` adapter would report a zero-norm gradient for every
//! `lora_a` and could not distinguish "the graph does not reach this
//! parameter" from "the initialisation makes the first step zero".
//! `Gaussian` gives both matrices a non-zero seeded draw, so a zero norm
//! can only mean an unreachable parameter — which is what the assertion is
//! for. The BERT-family oracles make the same choice for the same reason.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use jammi_encoders::{
    ClipText, ClipTextConfig, HtsatAudio, HtsatAudioConfig, OpenClipVisionConfig,
    OpenClipVisionTransformer,
};
use jammi_lora::{save_adapter, AdapterConfig, ComputePrecision, LoraBuildConfig, LoraInitMode};

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn open_clip_dir() -> PathBuf {
    root().join("tests/fixtures/tiny_open_clip")
}

fn htsat_dir() -> PathBuf {
    root().join("cookbook/fixtures/htsat_clap_tiny")
}

fn open_clip_json() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(open_clip_dir().join("open_clip_config.json")).unwrap(),
    )
    .unwrap()
}

fn htsat_json() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(htsat_dir().join("config.json")).unwrap())
        .unwrap()
}

/// The one `all-linear`, `Gaussian`, fixed-seed build config every A1/A5
/// oracle uses. `rank_pattern` empty, no dropout (a dropout stream would
/// add a second source of run-to-run variation this oracle does not need).
struct LoraFixture {
    targets: Vec<String>,
    layers: Option<Vec<usize>>,
    rank_pattern: HashMap<String, usize>,
}

impl LoraFixture {
    fn new(targets: &[&str]) -> Self {
        Self {
            targets: targets.iter().map(|s| s.to_string()).collect(),
            layers: None,
            rank_pattern: HashMap::new(),
        }
    }

    fn config(&self, init_mode: LoraInitMode) -> LoraBuildConfig<'_> {
        LoraBuildConfig {
            target_modules: &self.targets,
            layers_to_transform: &self.layers,
            lora_rank: 4,
            lora_alpha: 8.0,
            use_rslora: false,
            lora_dropout: None,
            rank_pattern: &self.rank_pattern,
            init_mode,
            seed: 0x5eed_1234,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assertions
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed non-uniform per-column loss weights: `Tensor::backward`'s implicit
/// seed gradient is `ones_like`, which could let a sign-flipped or
/// transposed reduction cancel to zero and read as "unreachable".
fn nonuniform_loss(out: &Tensor, device: &Device) -> Tensor {
    let cols = out.dim(1).unwrap();
    let w: Vec<f32> = (0..cols).map(|i| 1.0 + i as f32 * 0.37).collect();
    let w = Tensor::from_vec(w, cols, device)
        .unwrap()
        .to_dtype(out.dtype())
        .unwrap();
    out.broadcast_mul(&w).unwrap().sum_all().unwrap()
}

/// Every trainable param has a `Some`, FINITE, non-zero gradient.
///
/// Finiteness is asserted first and separately: `NaN > 0.0` is `false` (so a
/// bare `> 0.0` already fails on NaN) but `+inf > 0.0` is `true`, and an
/// exploded gradient would otherwise pass as a legitimate positive control.
fn assert_every_trainable_param_has_a_gradient(params: &[&Tensor], loss: &Tensor, what: &str) {
    assert!(
        !params.is_empty(),
        "{what}: the oracle is vacuous with zero trainable params"
    );
    let grads = loss.backward().unwrap();
    for (i, p) in params.iter().enumerate() {
        let g = grads
            .get(p)
            .unwrap_or_else(|| panic!("{what}: trainable param {i} has NO gradient entry"));
        let norm = g
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert!(
            norm.is_finite(),
            "{what}: trainable param {i} grad norm must be finite, got {norm}"
        );
        assert!(
            norm > 0.0,
            "{what}: trainable param {i} grad norm must be non-zero, got {norm}"
        );
    }
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-tower fixed batches
// ─────────────────────────────────────────────────────────────────────────────

fn clip_text_batch(device: &Device) -> (Tensor, Tensor) {
    let ids: Vec<u32> = vec![1, 5, 9, 13, 96, 0, 0, 0, 1, 2, 3, 4, 5, 6, 96, 0];
    (
        Tensor::from_vec(ids, (2, 8), device).unwrap(),
        Tensor::ones((2, 8), DType::U32, device).unwrap(),
    )
}

fn vision_batch(device: &Device, dtype: DType) -> Tensor {
    let n = 2 * 3 * 8 * 8;
    let px: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017 - 1.0).sin() * 0.5)
        .collect();
    Tensor::from_vec(px, (2, 3, 8, 8), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn htsat_batch(device: &Device, dtype: DType) -> Tensor {
    let pinned =
        candle_core::safetensors::load(htsat_dir().join("pinned_input.safetensors"), device)
            .unwrap();
    pinned
        .get("input_features")
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────
// A1 / A5: gradient reachability at F32 and F16
// ─────────────────────────────────────────────────────────────────────────────

/// Site count the CLIP block stack predicts: four LoRA sites per residual
/// block (`in_proj`, `out_proj`, `c_fc`, `c_proj`), two `Var`s each.
fn clip_expected_params(layers: usize) -> usize {
    2 * 4 * layers
}

/// Site count the HTSAT config predicts: six per Swin block, one
/// patch-merging `reduction` per stage EXCEPT the last, plus the two
/// projection-head linears; two `Var`s each.
fn htsat_expected_params(cfg: &HtsatAudioConfig) -> usize {
    let blocks: usize = cfg.depths.iter().sum();
    2 * (6 * blocks + (cfg.num_stages() - 1) + 2)
}

fn clip_text_reachability(backbone_dtype: DType) {
    let device = Device::Cpu;
    let cfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let fixture = LoraFixture::new(&["all-linear"]);
    let varmap = VarMap::new();
    let weights = open_clip_dir().join("open_clip_model.safetensors");
    let mut tower = ClipText::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .backbone_dtype(backbone_dtype)
        .build(&[weights.as_path()], &cfg, &device, &varmap)
        .unwrap();
    tower.set_training(true);
    assert!(tower.is_training());

    let params = tower.trainable_params();
    assert_eq!(
        params.len(),
        clip_expected_params(cfg.layers),
        "clip_text @ {backbone_dtype:?}: trainable Var count must be 2 x (4 sites x {} layers)",
        cfg.layers
    );

    let (input_ids, mask) = clip_text_batch(&device);
    let out = tower.forward(&input_ids, &mask).unwrap();
    assert_eq!(out.dtype(), backbone_dtype);
    let loss = nonuniform_loss(&out, &device);
    assert_every_trainable_param_has_a_gradient(
        &params,
        &loss,
        &format!("clip_text @ {backbone_dtype:?}"),
    );
}

fn open_clip_vision_reachability(backbone_dtype: DType) {
    let device = Device::Cpu;
    let cfg = OpenClipVisionConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let fixture = LoraFixture::new(&["all-linear"]);
    let varmap = VarMap::new();
    let weights = open_clip_dir().join("open_clip_model.safetensors");
    let mut tower = OpenClipVisionTransformer::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .backbone_dtype(backbone_dtype)
        .build(&[weights.as_path()], &cfg, &device, &varmap)
        .unwrap();
    tower.set_training(true);
    assert!(tower.is_training());

    let params = tower.trainable_params();
    assert_eq!(
        params.len(),
        clip_expected_params(cfg.layers),
        "open_clip_vision @ {backbone_dtype:?}: trainable Var count must be 2 x (4 sites x {} \
         layers)",
        cfg.layers
    );

    // A media tower's front end is a `Conv2d` (and, for HTSAT, a
    // `BatchNorm`): unlike `jammi_lora::FrozenBase::Dense`, candle's conv
    // does NOT cast its input to the weight dtype, so the caller supplies
    // the batch in the backbone's dtype.
    let pixels = vision_batch(&device, backbone_dtype);
    let out = tower.forward(&pixels).unwrap();
    assert_eq!(out.dtype(), backbone_dtype);
    let loss = nonuniform_loss(&out, &device);
    assert_every_trainable_param_has_a_gradient(
        &params,
        &loss,
        &format!("open_clip_vision @ {backbone_dtype:?}"),
    );
}

fn htsat_reachability(backbone_dtype: DType) {
    let device = Device::Cpu;
    let cfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let fixture = LoraFixture::new(&["all-linear"]);
    let varmap = VarMap::new();
    let weights = htsat_dir().join("model.safetensors");
    let mut tower = HtsatAudio::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .backbone_dtype(backbone_dtype)
        .build(&[weights.as_path()], &cfg, &device, &varmap)
        .unwrap();
    tower.set_training(true);
    assert!(tower.is_training());

    let params = tower.trainable_params();
    assert_eq!(
        params.len(),
        htsat_expected_params(&cfg),
        "htsat_audio @ {backbone_dtype:?}: trainable Var count must be 2 x (6 x sum(depths) + \
         (stages - 1) + 2)"
    );

    let feats = htsat_batch(&device, backbone_dtype);
    let out = tower.forward(&feats, &[true, true]).unwrap();
    assert_eq!(out.dtype(), backbone_dtype);
    let loss = nonuniform_loss(&out, &device);
    assert_every_trainable_param_has_a_gradient(
        &params,
        &loss,
        &format!("htsat_audio @ {backbone_dtype:?}"),
    );
}

#[test]
fn a1_clip_text_every_lora_param_is_reachable_at_f32() {
    clip_text_reachability(DType::F32);
}

#[test]
fn a1_open_clip_vision_every_lora_param_is_reachable_at_f32() {
    open_clip_vision_reachability(DType::F32);
}

#[test]
fn a1_htsat_every_lora_param_is_reachable_at_f32() {
    htsat_reachability(DType::F32);
}

/// A5. F16, on CPU. BF16 is deliberately absent: candle-core 0.11's CPU
/// matmul supports only F16/F32/F64, so a CPU bf16 forward cannot run at
/// all — the bf16 claim this unit makes is about MASK CONSTRUCTION only
/// (`clip_text`'s own `causal_mask_at_bf16_uses_the_bf16_minimum` unit
/// test), never about a CPU forward.
#[test]
fn a5_clip_text_every_lora_param_is_reachable_at_f16() {
    clip_text_reachability(DType::F16);
}

/// See [`a5_clip_text_every_lora_param_is_reachable_at_f16`].
#[test]
fn a5_open_clip_vision_every_lora_param_is_reachable_at_f16() {
    open_clip_vision_reachability(DType::F16);
}

/// See [`a5_clip_text_every_lora_param_is_reachable_at_f16`].
#[test]
fn a5_htsat_every_lora_param_is_reachable_at_f16() {
    htsat_reachability(DType::F16);
}

// ─────────────────────────────────────────────────────────────────────────────
// A2: eval bit-identity
// ─────────────────────────────────────────────────────────────────────────────

/// A2 (i) and (ii) for all three towers in one place, since the three
/// baselines are the same claim about three different loaders:
/// `builder(frozen()) == load()`, and `ZerosB` on `all-linear` == unadapted.
///
/// `ZerosB` sets `B = 0`, so the LoRA branch contributes exactly `0.0` and
/// the epilogue's add is an exact identity — an installed-but-zero adapter
/// must not perturb a single output bit, which is what makes "load an
/// adapter and serve" safe to do before any training has happened.
#[test]
fn a2_builder_frozen_and_zerosb_adapter_are_bit_identical_to_load() {
    let device = Device::Cpu;
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let frozen = LoraFixture::new(&[]);
    let all_linear = LoraFixture::new(&["all-linear"]);

    // CLIP text.
    let tcfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let (ids, mask) = clip_text_batch(&device);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&ocp), DType::F32, &device)
            .unwrap()
    };
    let loaded = ClipText::load(vb.clone(), &tcfg).unwrap();
    let reference = bits(&loaded.forward(&ids, &mask).unwrap());

    let varmap = VarMap::new();
    let built = ClipText::builder()
        .lora(frozen.config(LoraInitMode::ZerosB))
        .build(&[ocp.as_path()], &tcfg, &device, &varmap)
        .unwrap();
    assert!(built.trainable_params().is_empty());
    assert_eq!(
        bits(&built.forward(&ids, &mask).unwrap()),
        reference,
        "clip_text: builder(frozen()) must be bit-identical to load()"
    );

    let varmap = VarMap::new();
    let zeros_b = ClipText::builder()
        .lora(all_linear.config(LoraInitMode::ZerosB))
        .build(&[ocp.as_path()], &tcfg, &device, &varmap)
        .unwrap();
    assert_eq!(
        zeros_b.trainable_params().len(),
        clip_expected_params(tcfg.layers)
    );
    assert_eq!(
        bits(&zeros_b.forward(&ids, &mask).unwrap()),
        reference,
        "clip_text: a ZerosB adapter on every site must not move a single output bit"
    );

    // OpenCLIP vision.
    let vcfg = OpenClipVisionConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let pixels = vision_batch(&device, DType::F32);
    let loaded = OpenClipVisionTransformer::load(vb.pp("visual"), &vcfg).unwrap();
    let reference = bits(&loaded.forward(&pixels).unwrap());

    let varmap = VarMap::new();
    let built = OpenClipVisionTransformer::builder()
        .lora(frozen.config(LoraInitMode::ZerosB))
        .build(&[ocp.as_path()], &vcfg, &device, &varmap)
        .unwrap();
    assert_eq!(
        bits(&built.forward(&pixels).unwrap()),
        reference,
        "open_clip_vision: builder(frozen()) must be bit-identical to load()"
    );

    let varmap = VarMap::new();
    let zeros_b = OpenClipVisionTransformer::builder()
        .lora(all_linear.config(LoraInitMode::ZerosB))
        .build(&[ocp.as_path()], &vcfg, &device, &varmap)
        .unwrap();
    assert_eq!(
        bits(&zeros_b.forward(&pixels).unwrap()),
        reference,
        "open_clip_vision: a ZerosB adapter on every site must not move a single output bit"
    );

    // HTSAT audio.
    let acfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let hp = htsat_dir().join("model.safetensors");
    let feats = htsat_batch(&device, DType::F32);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&hp), DType::F32, &device).unwrap()
    };
    let loaded = HtsatAudio::load(vb, &acfg, &device).unwrap();
    let reference = bits(&loaded.forward(&feats, &[true, true]).unwrap());

    let varmap = VarMap::new();
    let built = HtsatAudio::builder()
        .lora(frozen.config(LoraInitMode::ZerosB))
        .build(&[hp.as_path()], &acfg, &device, &varmap)
        .unwrap();
    assert_eq!(
        bits(&built.forward(&feats, &[true, true]).unwrap()),
        reference,
        "htsat_audio: builder(frozen()) must be bit-identical to load()"
    );

    let varmap = VarMap::new();
    let zeros_b = HtsatAudio::builder()
        .lora(all_linear.config(LoraInitMode::ZerosB))
        .build(&[hp.as_path()], &acfg, &device, &varmap)
        .unwrap();
    assert_eq!(
        zeros_b.trainable_params().len(),
        htsat_expected_params(&acfg)
    );
    assert_eq!(
        bits(&zeros_b.forward(&feats, &[true, true]).unwrap()),
        reference,
        "htsat_audio: a ZerosB adapter on every site must not move a single output bit"
    );
}

/// Negative control for the A2 oracle above: a GAUSSIAN adapter (non-zero
/// `B`) on the same sites DOES move the output. Without this, "ZerosB
/// changes nothing" would also pass if the builder silently dropped the
/// adapter entirely.
#[test]
fn a2_control_a_gaussian_adapter_does_change_the_output() {
    let device = Device::Cpu;
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let tcfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let (ids, mask) = clip_text_batch(&device);
    let all_linear = LoraFixture::new(&["all-linear"]);

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&ocp), DType::F32, &device)
            .unwrap()
    };
    let reference = bits(
        &ClipText::load(vb, &tcfg)
            .unwrap()
            .forward(&ids, &mask)
            .unwrap(),
    );

    let varmap = VarMap::new();
    let adapted = ClipText::builder()
        .lora(all_linear.config(LoraInitMode::Gaussian))
        .build(&[ocp.as_path()], &tcfg, &device, &varmap)
        .unwrap();
    assert_ne!(
        bits(&adapted.forward(&ids, &mask).unwrap()),
        reference,
        "a Gaussian adapter must actually be installed and change the output"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// A7: probe input
// ─────────────────────────────────────────────────────────────────────────────

/// A7. `AnyEncoder::probe_input` yields a batch each tower's own
/// `forward_input` accepts — the smallest VALID one for that variant's
/// geometry, derived from the encoder itself rather than a caller-side
/// table.
#[test]
fn a7_probe_input_forward_succeeds_on_every_tower() {
    use jammi_encoders::AnyEncoder;
    let device = Device::Cpu;
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&ocp), DType::F32, &device)
            .unwrap()
    };

    let tcfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let vcfg = OpenClipVisionConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let acfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let hvb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[htsat_dir().join("model.safetensors")],
            DType::F32,
            &device,
        )
        .unwrap()
    };

    let encoders = [
        (
            "clip_text",
            AnyEncoder::ClipText(ClipText::load(vb.clone(), &tcfg).unwrap()),
            tcfg.embed_dim,
        ),
        (
            "open_clip_vision",
            AnyEncoder::OpenClipVision(
                OpenClipVisionTransformer::load(vb.pp("visual"), &vcfg).unwrap(),
            ),
            vcfg.embed_dim,
        ),
        (
            "htsat_audio",
            AnyEncoder::Htsat(Box::new(HtsatAudio::load(hvb, &acfg, &device).unwrap())),
            acfg.projection_dim,
        ),
    ];

    for (name, encoder, out_dim) in encoders {
        let probe = encoder.probe_input(&device).unwrap();
        assert_eq!(
            probe.modality(),
            encoder.modality(),
            "{name}: the probe batch must be for the encoder's OWN modality"
        );
        let out = encoder
            .forward_input(&probe.as_input())
            .unwrap_or_else(|e| panic!("{name}: probe forward failed: {e}"));
        assert_eq!(
            out.dims(),
            &[1, out_dim],
            "{name}: probe output must be [1, output_dim]"
        );
        assert_eq!(encoder.hidden_size(), out_dim);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adapter round trip
// ─────────────────────────────────────────────────────────────────────────────

/// Train (well: install a Gaussian adapter and perturb it), export through
/// `named_trainable_weights`, persist with `jammi_lora::save_adapter`,
/// rebuild through `builder().adapter(Some(path))`, and require the rebuilt
/// tower's EVAL output to be bit-equal to the source tower's.
///
/// This is the property a served fine-tune depends on: what the trainer had
/// in memory is exactly what the serving path reconstructs. A key-layout
/// mismatch between `named_trainable_weights` and the builder's trainable
/// `VarBuilder` scoping would show up here as `lora_a` silently falling back
/// to its `Init::Const(0.0)` default — i.e. a different (unadapted) output —
/// rather than as an error.
fn adapter_round_trip<T, B, F>(name: &str, build: B, forward: F, model_type: &str)
where
    B: Fn(Option<&Path>, &VarMap, LoraBuildConfig<'_>) -> T,
    F: Fn(&T) -> Tensor,
    T: TowerHooks,
{
    let dir = tempfile::tempdir().unwrap();
    let fixture = LoraFixture::new(&["all-linear"]);

    let varmap = VarMap::new();
    let trained = build(None, &varmap, fixture.config(LoraInitMode::Gaussian));
    let weights = trained.named_weights();
    assert_eq!(
        weights.len(),
        trained.param_count(),
        "{name}: every trainable Var must be exported by name"
    );
    let reference = bits(&forward(&trained));

    let cfg = AdapterConfig::from_build(
        model_type,
        &fixture.config(LoraInitMode::Gaussian),
        ComputePrecision::F32,
    );
    save_adapter(dir.path(), &weights, &cfg).unwrap();

    let varmap2 = VarMap::new();
    let restored = build(
        Some(&dir.path().join("adapter.safetensors")),
        &varmap2,
        fixture.config(LoraInitMode::Gaussian),
    );
    assert_eq!(
        bits(&forward(&restored)),
        reference,
        "{name}: a tower rebuilt from the saved adapter must be bit-equal at eval"
    );
}

/// The two hooks [`adapter_round_trip`] needs from a tower, so the helper
/// is generic over the three rather than copied three times.
trait TowerHooks {
    fn named_weights(&self) -> HashMap<String, Tensor>;
    fn param_count(&self) -> usize;
}

impl TowerHooks for ClipText {
    fn named_weights(&self) -> HashMap<String, Tensor> {
        self.named_trainable_weights().unwrap()
    }
    fn param_count(&self) -> usize {
        self.trainable_params().len()
    }
}

impl TowerHooks for OpenClipVisionTransformer {
    fn named_weights(&self) -> HashMap<String, Tensor> {
        self.named_trainable_weights().unwrap()
    }
    fn param_count(&self) -> usize {
        self.trainable_params().len()
    }
}

impl TowerHooks for HtsatAudio {
    fn named_weights(&self) -> HashMap<String, Tensor> {
        self.named_trainable_weights().unwrap()
    }
    fn param_count(&self) -> usize {
        self.trainable_params().len()
    }
}

#[test]
fn adapter_round_trip_clip_text() {
    let device = Device::Cpu;
    let cfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let (ids, mask) = clip_text_batch(&device);
    adapter_round_trip(
        "clip_text",
        |adapter, varmap, lora| {
            ClipText::builder()
                .lora(lora)
                .adapter(adapter)
                .build(&[ocp.as_path()], &cfg, &device, varmap)
                .unwrap()
        },
        |m| m.forward(&ids, &mask).unwrap(),
        "open_clip",
    );
}

#[test]
fn adapter_round_trip_open_clip_vision() {
    let device = Device::Cpu;
    let cfg = OpenClipVisionConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let pixels = vision_batch(&device, DType::F32);
    adapter_round_trip(
        "open_clip_vision",
        |adapter, varmap, lora| {
            OpenClipVisionTransformer::builder()
                .lora(lora)
                .adapter(adapter)
                .build(&[ocp.as_path()], &cfg, &device, varmap)
                .unwrap()
        },
        |m| m.forward(&pixels).unwrap(),
        "open_clip",
    );
}

#[test]
fn adapter_round_trip_htsat_audio() {
    let device = Device::Cpu;
    let cfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let hp = htsat_dir().join("model.safetensors");
    let feats = htsat_batch(&device, DType::F32);
    adapter_round_trip(
        "htsat_audio",
        |adapter, varmap, lora| {
            HtsatAudio::builder()
                .lora(lora)
                .adapter(adapter)
                .build(&[hp.as_path()], &cfg, &device, varmap)
                .unwrap()
        },
        |m| m.forward(&feats, &[true, true]).unwrap(),
        "clap_audio_model",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Selector semantics
// ─────────────────────────────────────────────────────────────────────────────

/// A selector that names no real site of this tower installs NOTHING — it
/// does not silently fall back to `all-linear`. The zero-trainable outcome
/// is what a caller's own "no trainable parameters" refusal keys on, so it
/// must be reachable and honest.
#[test]
fn a_selector_that_matches_no_site_installs_no_adapter() {
    let device = Device::Cpu;
    let cfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    // `q_proj` is a BERT/LLaMA-family name; the OpenCLIP block's own sites
    // are `in_proj`, `out_proj`, `c_fc`, `c_proj`.
    let fixture = LoraFixture::new(&["q_proj"]);
    let varmap = VarMap::new();
    let tower = ClipText::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .build(&[ocp.as_path()], &cfg, &device, &varmap)
        .unwrap();
    assert!(tower.trainable_params().is_empty());
    assert!(tower.named_trainable_weights().unwrap().is_empty());

    // Positive control on the SAME tower: one of its real site names does
    // install adapters, so the emptiness above is the selector's doing.
    let fixture = LoraFixture::new(&["c_fc"]);
    let varmap = VarMap::new();
    let tower = ClipText::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .build(&[ocp.as_path()], &cfg, &device, &varmap)
        .unwrap();
    assert_eq!(tower.trainable_params().len(), 2 * cfg.layers);
}

/// `layers_to_transform` restricts by STAGE on HTSAT (PEFT's first-numbered-
/// segment rule) and, being an active layer filter, excludes the UNINDEXED
/// `audio_projection` head entirely — the concrete consequence of
/// `should_apply_lora`'s `(None, Some(_)) => false` arm.
#[test]
fn htsat_layers_to_transform_selects_stages_and_excludes_the_unindexed_head() {
    let device = Device::Cpu;
    let cfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let hp = htsat_dir().join("model.safetensors");
    let targets: Vec<String> = vec!["all-linear".into()];
    let layers = Some(vec![0usize]);
    let rank_pattern = HashMap::new();
    let lora = LoraBuildConfig {
        target_modules: &targets,
        layers_to_transform: &layers,
        lora_rank: 4,
        lora_alpha: 8.0,
        use_rslora: false,
        lora_dropout: None,
        rank_pattern: &rank_pattern,
        init_mode: LoraInitMode::Gaussian,
        seed: 7,
    };
    let varmap = VarMap::new();
    let tower = HtsatAudio::builder()
        .lora(lora)
        .build(&[hp.as_path()], &cfg, &device, &varmap)
        .unwrap();

    let keys: Vec<String> = tower
        .named_trainable_weights()
        .unwrap()
        .into_keys()
        .collect();
    assert!(
        keys.iter().all(|k| k.starts_with("layers.0.")),
        "only stage 0 sites may be adapted, got {keys:?}"
    );
    assert!(
        !keys.iter().any(|k| k.starts_with("audio_projection")),
        "the unindexed projection head must be excluded by an active layer filter, got {keys:?}"
    );
    // Stage 0 has `depths[0]` blocks x 6 sites, plus its downsample.
    let expected = 2 * (6 * cfg.depths[0] + 1);
    assert_eq!(tower.trainable_params().len(), expected);
}

// ─────────────────────────────────────────────────────────────────────────────
// Adapter key layout
// ─────────────────────────────────────────────────────────────────────────────

/// The adapter key set is EXACT, not merely "contains what we looked for".
///
/// These strings are the wire format a saved adapter carries, so a serving
/// path reconstructs a tower by matching them; a silent rename would not
/// error at load (a missing `lora_a` falls back to `Init::Const(0.0)`), it
/// would serve an unadapted model. Every key is built from the checkpoint's
/// OWN path vocabulary (`resblocks.{n}`, `layers.{s}.blocks.{b}`,
/// `layers.{s}.downsample`, `audio_projection`) plus the site leaf.
#[test]
fn adapter_key_layout_is_exactly_the_checkpoint_shaped_set() {
    let device = Device::Cpu;
    let fixture = LoraFixture::new(&["all-linear"]);

    // CLIP text: 4 sites x layers.
    let tcfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let varmap = VarMap::new();
    let text = ClipText::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .build(&[ocp.as_path()], &tcfg, &device, &varmap)
        .unwrap();
    let mut got: Vec<String> = text
        .named_trainable_weights()
        .unwrap()
        .into_keys()
        .collect();
    got.sort();
    let mut want: Vec<String> = Vec::new();
    for n in 0..tcfg.layers {
        for site in ["in_proj", "out_proj", "c_fc", "c_proj"] {
            for ab in ["lora_a", "lora_b"] {
                want.push(format!("resblocks.{n}.{site}.{ab}"));
            }
        }
    }
    want.sort();
    assert_eq!(got, want, "clip_text adapter key set");

    // HTSAT: 6 per block, one reduction per non-final stage, two head sites.
    let acfg = HtsatAudioConfig::from_hf_clap_config(&htsat_json()).unwrap();
    let hp = htsat_dir().join("model.safetensors");
    let varmap = VarMap::new();
    let audio = HtsatAudio::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .build(&[hp.as_path()], &acfg, &device, &varmap)
        .unwrap();
    let mut got: Vec<String> = audio
        .named_trainable_weights()
        .unwrap()
        .into_keys()
        .collect();
    got.sort();
    let mut want: Vec<String> = Vec::new();
    for (s, &depth) in acfg.depths.iter().enumerate() {
        for b in 0..depth {
            for site in [
                "query",
                "key",
                "value",
                "attention_output",
                "intermediate_dense",
                "output_dense",
            ] {
                for ab in ["lora_a", "lora_b"] {
                    want.push(format!("layers.{s}.blocks.{b}.{site}.{ab}"));
                }
            }
        }
        if s + 1 < acfg.num_stages() {
            for ab in ["lora_a", "lora_b"] {
                want.push(format!("layers.{s}.downsample.reduction.{ab}"));
            }
        }
    }
    for site in ["linear1", "linear2"] {
        for ab in ["lora_a", "lora_b"] {
            want.push(format!("audio_projection.{site}.{ab}"));
        }
    }
    want.sort();
    assert_eq!(got, want, "htsat_audio adapter key set");
}

/// D1's fused-QKV claim, measured on the adapter tensors themselves: the
/// OpenCLIP block's `in_proj` is ONE LoRA site over the full `[3*width,
/// width]` fused projection, not three sites of `[width, width]`. Its
/// `lora_a` is `[rank, width]` and its `lora_b` is `[3*width, rank]`, so a
/// single adapter jointly adapts Q, K and V — the layout PEFT itself uses
/// for `nn.MultiheadAttention.in_proj_weight`.
#[test]
fn the_fused_qkv_site_is_one_adapter_over_three_times_width() {
    let device = Device::Cpu;
    let cfg = ClipTextConfig::from_open_clip_config(&open_clip_json()).unwrap();
    let ocp = open_clip_dir().join("open_clip_model.safetensors");
    let fixture = LoraFixture::new(&["in_proj"]);
    let rank = fixture.config(LoraInitMode::Gaussian).lora_rank;
    let varmap = VarMap::new();
    let tower = ClipText::builder()
        .lora(fixture.config(LoraInitMode::Gaussian))
        .build(&[ocp.as_path()], &cfg, &device, &varmap)
        .unwrap();

    let weights = tower.named_trainable_weights().unwrap();
    assert_eq!(
        weights.len(),
        2 * cfg.layers,
        "the `in_proj` selector must match exactly one site per block"
    );
    let a = weights.get("resblocks.0.in_proj.lora_a").unwrap();
    let b = weights.get("resblocks.0.in_proj.lora_b").unwrap();
    assert_eq!(a.dims(), &[rank, cfg.width]);
    assert_eq!(
        b.dims(),
        &[3 * cfg.width, rank],
        "lora_b must span the FUSED 3*width output, proving Q/K/V share one adapter"
    );
}
