//! Golden-reference parity harness for the HTSAT-Swin CLAP audio tower.
//!
//! `candle-transformers` has no CLAP/Swin/HTSAT module, so this port has no
//! in-Candle reference to parity-test against (unlike `tests/parity.rs`, which
//! checks `jammi_encoders::Bert` against `candle-transformers`). Instead the
//! oracle is a set of committed per-boundary golden activations dumped from the
//! real PyTorch `transformers` `ClapAudioModelWithProjection` by
//! `tests/fixtures/generate_htsat_clap.py`. Every unit of the Rust tower is
//! asserted against its golden boundary, so a divergence localizes to the unit
//! that produced it.
//!
//! Gated behind the `golden-parity` feature so the default `cargo test` stays
//! free of the committed-golden machinery. The goldens are committed binaries;
//! the feature needs no torch and makes no network call.
//!
//! Two tolerance metrics back the boundary assertions: max-abs for
//! large-magnitude intermediates, and cosine for the final L2-normalized
//! embedding (whose unnormalized norm is small enough that max-abs is
//! dishonest). `goldens_are_self_consistent` checks the committed goldens'
//! internal consistency independently of any tower.
#![cfg(feature = "golden-parity")]

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use jammi_encoders::htsat_audio::{HtsatAudioConfig, HtsatAudioEncoder};

/// Max-abs tolerance for large-magnitude intermediate boundaries. Looser than
/// `parity.rs`'s 1e-5 because the oracle is a *different* framework (PyTorch)
/// vs Candle — accumulated fp32 kernel-order differences are larger than the
/// same-framework BERT parity — yet strict enough that a wrong axis, scale, or
/// activation (which diverges by >=1e-2) cannot pass. Never loosened per-test.
const TOL_ABS: f32 = 1e-4;

/// Cosine floor for the final normalized embedding. Max-abs is dishonest there:
/// the unnormalized projection has norm ~0.02, so L2-normalizing amplifies tiny
/// absolute perturbations ~40x. Direction is what the embedding contract means,
/// so cosine similarity is the faithful metric.
const MIN_COS: f32 = 1.0 - 1e-5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/htsat_clap_tiny")
}

/// The committed per-boundary golden activations, loaded once.
struct Goldens {
    tensors: HashMap<String, Tensor>,
}

impl Goldens {
    fn load() -> candle_core::Result<Self> {
        let path = fixture_dir().join("goldens.safetensors");
        let tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
        Ok(Self { tensors })
    }

    /// Borrow the golden tensor for `name`, or an error naming what is missing.
    fn get(&self, name: &str) -> candle_core::Result<&Tensor> {
        self.tensors.get(name).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "golden tensor '{name}' not found in {}",
                fixture_dir().join("goldens.safetensors").display()
            ))
        })
    }
}

/// The deterministic `input_features` the tower is tested on, independent of
/// the audio front-end.
fn load_pinned_input() -> candle_core::Result<Tensor> {
    let path = fixture_dir().join("pinned_input.safetensors");
    let mut tensors = candle_core::safetensors::load(&path, &Device::Cpu)?;
    tensors.remove("input_features").ok_or_else(|| {
        candle_core::Error::Msg(format!("'input_features' not found in {}", path.display()))
    })
}

/// Max absolute element difference between two equally-shaped tensors.
fn max_abs_diff(got: &Tensor, golden: &Tensor) -> candle_core::Result<f32> {
    (got - golden)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

/// Minimum per-row cosine similarity (over the batch) between two `[B, D]`
/// tensors.
fn row_cosine_min(got: &Tensor, golden: &Tensor) -> candle_core::Result<f32> {
    let dot = (got * golden)?.sum(D::Minus1)?;
    let n_got = got.sqr()?.sum(D::Minus1)?.sqrt()?;
    let n_golden = golden.sqr()?.sum(D::Minus1)?.sqrt()?;
    let cos = (dot / (n_got * n_golden)?)?;
    cos.min(0)?.to_scalar::<f32>()
}

/// Assert a tower tensor matches its golden within the strict max-abs
/// tolerance. For large-magnitude intermediate boundaries.
fn assert_close_max_abs(got: &Tensor, golden: &Tensor, name: &str) {
    assert_eq!(got.dims(), golden.dims(), "{name}: shape mismatch");
    let diff = max_abs_diff(got, golden).expect("compute max-abs diff");
    assert!(diff < TOL_ABS, "{name}: max |Δ| = {diff} >= {TOL_ABS}");
}

/// Assert a tower embedding matches its golden by direction (cosine). For the
/// final L2-normalized embedding.
fn assert_close_cosine(got: &Tensor, golden: &Tensor, name: &str) {
    assert_eq!(got.dims(), golden.dims(), "{name}: shape mismatch");
    let cos = row_cosine_min(got, golden).expect("compute row cosine");
    assert!(cos >= MIN_COS, "{name}: min row cosine = {cos} < {MIN_COS}");
}

/// Prove the loader, both metrics, and the committed goldens are
/// self-consistent, independently of any tower. A corrupt or mis-regenerated
/// fixture fails here.
#[test]
fn goldens_are_self_consistent() {
    let goldens = Goldens::load().expect("load goldens.safetensors");

    // Every golden's shape matches the committed manifest (catches a fixture
    // regenerated with drifted geometry).
    let manifest_str = std::fs::read_to_string(fixture_dir().join("golden_manifest.json"))
        .expect("read golden_manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_str).expect("parse golden_manifest.json");
    let manifest = manifest.as_object().expect("manifest is an object");
    assert!(!manifest.is_empty(), "manifest has no entries");
    for (name, spec) in manifest {
        let want: Vec<usize> = spec["shape"]
            .as_array()
            .expect("shape is an array")
            .iter()
            .map(|d| d.as_u64().expect("shape dim is u64") as usize)
            .collect();
        let tensor = goldens
            .get(name)
            .expect("golden present for manifest entry");
        assert_eq!(
            tensor.dims(),
            want.as_slice(),
            "{name}: manifest shape mismatch"
        );
    }

    // The normalized embedding is exactly the L2-normalization of the
    // unnormalized projection — the contract boundary the tower targets.
    let unnorm = goldens
        .get("projected_unnormalized")
        .expect("unnormalized golden");
    let norm = goldens
        .get("projected_normalized")
        .expect("normalized golden");
    let recomputed = unnorm
        .broadcast_div(
            &unnorm
                .sqr()
                .unwrap()
                .sum_keepdim(D::Minus1)
                .unwrap()
                .sqrt()
                .unwrap(),
        )
        .expect("recompute normalize");
    assert_close_cosine(
        &recomputed,
        norm,
        "projected_normalized vs normalize(unnormalized)",
    );

    // Normalized rows are unit length.
    let row_norms = norm.sqr().unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
    let min_norm = row_norms.min(0).unwrap().to_scalar::<f32>().unwrap();
    let max_norm = row_norms.max(0).unwrap().to_scalar::<f32>().unwrap();
    assert!(
        (min_norm - 1.0).abs() < TOL_ABS && (max_norm - 1.0).abs() < TOL_ABS,
        "normalized rows not unit length: [{min_norm}, {max_norm}]"
    );

    // The pinned input is the forward's `mel_in` boundary: the 4-channel
    // is_longer=True fused input the real checkpoint takes, with a time dim (500)
    // that triggers bicubic interpolation up to spec_width (512).
    let pinned = load_pinned_input().expect("load pinned_input");
    assert_eq!(pinned.dims(), &[2, 4, 500, 32], "pinned input shape");
    let mel_in = goldens.get("mel_in").expect("mel_in golden");
    assert_close_max_abs(&pinned, mel_in, "mel_in vs pinned_input");
}

/// Parse the committed fixture `config.json` into an [`HtsatAudioConfig`].
fn load_config() -> HtsatAudioConfig {
    let path = fixture_dir().join("config.json");
    let s = std::fs::read_to_string(&path).expect("read config.json");
    let json: serde_json::Value = serde_json::from_str(&s).expect("parse config.json");
    HtsatAudioConfig::from_hf_clap_config(&json).expect("HtsatAudioConfig from fixture")
}

/// Build the front-half encoder from the committed `model.safetensors`.
fn load_front_encoder(device: &Device) -> HtsatAudioEncoder {
    let config = load_config();
    let weights = fixture_dir().join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)
            .expect("mmap model.safetensors")
    };
    let encoder_vb = vb.pp("audio_model").pp("audio_encoder");
    HtsatAudioEncoder::load(encoder_vb, &config, device).expect("load front-half encoder")
}

/// Drive the real front half of the HTSAT-Swin tower on the pinned input and
/// gate every boundary — batch-norm, bicubic time-resample, `reshape_mel2img`,
/// the rectangular-stride `mel_conv2d`, AFF fusion, and the flattened
/// `patch_embed` output — against its committed golden. A divergence localizes
/// to the unit that produced it.
#[test]
fn front_half_matches_goldens() {
    let device = Device::Cpu;
    let goldens = Goldens::load().expect("load goldens.safetensors");
    let encoder = load_front_encoder(&device);
    let input = load_pinned_input().expect("load pinned_input");

    let front = encoder.forward_front(&input).expect("front-half forward");

    assert_close_max_abs(
        &front.post_batch_norm,
        goldens.get("post_batch_norm").expect("post_batch_norm"),
        "post_batch_norm",
    );
    assert_close_max_abs(
        &front.post_interpolation,
        goldens
            .get("post_interpolation")
            .expect("post_interpolation"),
        "post_interpolation",
    );
    assert_close_max_abs(
        &front.post_reshape_mel2img,
        goldens
            .get("post_reshape_mel2img")
            .expect("post_reshape_mel2img"),
        "post_reshape_mel2img",
    );
    assert_close_max_abs(
        &front.mel_conv2d_out,
        goldens.get("mel_conv2d_out").expect("mel_conv2d_out"),
        "mel_conv2d_out",
    );
    assert_close_max_abs(
        &front.fusion_model_out,
        goldens.get("fusion_model_out").expect("fusion_model_out"),
        "fusion_model_out",
    );
    assert_close_max_abs(
        &front.patch_embed_out,
        goldens.get("patch_embed_out").expect("patch_embed_out"),
        "patch_embed_out",
    );
}
