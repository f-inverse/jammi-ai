//! Live numerical-acceptance harness for the HTSAT-Swin CLAP audio tower against
//! the REAL `laion/clap-htsat-fused` checkpoint.
//!
//! `tests/golden_parity.rs` proves the tower is bit-faithful to a *tiny*
//! randomly-initialized HTSAT at every boundary; this test proves the SAME tower
//! reproduces the *real* checkpoint's published audio embedding. It downloads the
//! real `model.safetensors` + `config.json` from the HF Hub, parses the config
//! into [`HtsatAudioConfig`], loads the tower, and runs it on the committed
//! `input_features` pinned input — the exact `[1, 4, 1001, 64]` fusion
//! spectrogram `transformers` produced for a deterministic 5s clip — asserting
//! the L2-normalized output matches the committed `embedding`
//! (`ClapModel.get_audio_features` on the feature extractor's natural output,
//! whose committed `is_longer` is the canonical always-fusion `true`) by
//! direction.
//!
//! Because the input is the committed `input_features` (not re-derived from
//! audio), this isolates the *tower*: there is no front-end in the loop, so a
//! divergence here is a tower-vs-real-model algorithmic or weight-loading bug,
//! not a DSP gap. The cosine floor is derived from the measured agreement.
//!
//! Gated behind `live-hub-tests` so the default `cargo test` makes no network
//! call.
#![cfg(feature = "live-hub-tests")]

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use jammi_encoders::htsat_audio::{HtsatAudio, HtsatAudioConfig};

const REAL_MODEL_ID: &str = "laion/clap-htsat-fused";

/// Cosine floor for the tower's embedding vs the real checkpoint's published
/// embedding, on the committed pinned `input_features`.
///
/// MEASURED min row cosine on this box: 0.99999994 (fp32 cosine, directions
/// identical to within fp32 rounding) against the canonical fusion golden. The
/// tiny full-tower golden test passes at 1 − 1e-5; the real tower is far deeper
/// (depths [2,2,6,2], hidden 768) yet the agreement is exact, because the input
/// is the SAME committed `input_features` AND the SAME committed `is_longer=true`
/// flag the golden's `get_audio_features` ran with — there is no
/// algorithmic/front-end error in the loop, only fp32 kernel-order rounding vs
/// the PyTorch reference. The floor is set to 1 − 1e-5, strict enough that any
/// real tower-vs-model divergence (wrong axis/scale/weight, or the wrong
/// patch-embed gate, which moves cosine by ≥1e-3) fails, with ample margin for
/// cross-host fp variation. Never loosened to pass.
const MIN_COS_REAL: f32 = 1.0 - 1e-5;

fn real_fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../cookbook/fixtures/htsat_clap_real")
}

/// Minimum per-row cosine similarity (over the batch) between two `[B, D]`
/// tensors. Direction is the embedding contract; max-abs is dishonest on a
/// unit-norm vector.
fn row_cosine_min(got: &Tensor, golden: &Tensor) -> candle_core::Result<f32> {
    let dot = (got * golden)?.sum(D::Minus1)?;
    let n_got = got.sqr()?.sum(D::Minus1)?.sqrt()?;
    let n_golden = golden.sqr()?.sum(D::Minus1)?.sqrt()?;
    let cos = (dot / (n_got * n_golden)?)?;
    cos.min(0)?.to_scalar::<f32>()
}

/// Resolve the real checkpoint's weights + config from the HF Hub (cached
/// on-disk after the first download), mirroring the `hf_hub` pattern in
/// `crates/jammi-ai/tests/it/models.rs`.
fn fetch_real_model() -> (std::path::PathBuf, std::path::PathBuf) {
    let api = hf_hub::api::sync::Api::new().expect("build hf_hub api");
    let repo = api.model(REAL_MODEL_ID.to_string());
    let weights = repo
        .get("model.safetensors")
        .expect("fetch model.safetensors");
    let config = repo.get("config.json").expect("fetch config.json");
    (weights, config)
}

#[test]
fn tower_matches_real_clap_embedding_on_pinned_input() {
    let device = Device::Cpu;
    let (weights_path, config_path) = fetch_real_model();

    // Parse the top-level ClapConfig's `audio_config` into HtsatAudioConfig.
    let config_str = std::fs::read_to_string(&config_path).expect("read config.json");
    let config_json: serde_json::Value =
        serde_json::from_str(&config_str).expect("parse config.json");
    let config =
        HtsatAudioConfig::from_hf_clap_config(&config_json).expect("HtsatAudioConfig from real");

    // Load the full tower (encoder + projection) from the real checkpoint root.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            .expect("mmap real model.safetensors")
    };
    let tower = HtsatAudio::load(vb, &config, &device).expect("load real HTSAT tower");

    // Committed golden: the pinned `input_features`, the matching per-clip
    // `is_longer` flag, and the real checkpoint's published `embedding`.
    let goldens =
        candle_core::safetensors::load(real_fixture_dir().join("goldens.safetensors"), &device)
            .expect("load real goldens.safetensors");
    let input_features = goldens
        .get("input_features")
        .expect("input_features golden")
        .clone();
    let embedding = goldens.get("embedding").expect("embedding golden").clone();

    // is_longer [1, 1] bool tensor -> Vec<bool> per clip.
    let is_longer_tensor = goldens.get("is_longer").expect("is_longer golden");
    let is_longer: Vec<bool> = is_longer_tensor
        .flatten_all()
        .expect("flatten is_longer")
        .to_vec1::<u8>()
        .expect("is_longer to u8")
        .into_iter()
        .map(|v| v != 0)
        .collect();

    assert_eq!(
        input_features.dims(),
        &[1, 4, 1001, 64],
        "pinned input_features shape"
    );
    assert_eq!(embedding.dims(), &[1, 512], "embedding shape");
    // Canonical always-fusion: the golden embedding is `get_audio_features` on
    // the feature extractor's natural output, whose `is_longer` is true (the
    // deterministic fusion promotion). jammi's front-end emits the same true
    // flag for every clip, so the tower runs the AFF fusion path here too.
    assert_eq!(
        is_longer,
        vec![true],
        "canonical always-fusion is_longer flag"
    );

    let got = tower
        .forward(&input_features, &is_longer)
        .expect("real tower forward");
    assert_eq!(got.dims(), &[1, 512], "tower output shape");

    let cos = row_cosine_min(&got, &embedding).expect("compute row cosine");
    assert!(
        cos >= MIN_COS_REAL,
        "tower vs real `laion/clap-htsat-fused` embedding: min row cosine = {cos} < {MIN_COS_REAL} \
         — a tower-vs-real-model divergence, not a tolerance issue"
    );
}
