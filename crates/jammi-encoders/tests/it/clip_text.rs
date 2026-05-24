//! OpenCLIP text encoder integration tests against the `tiny_open_clip`
//! fixture. Verifies the text tower loads from the same checkpoint root as
//! the vision tower (vision under `visual.*`, text at root) and produces
//! L2-normalized embeddings in the shared CLIP latent space.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use jammi_encoders::{ClipText, ClipTextConfig};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny_open_clip")
}

fn load_config() -> ClipTextConfig {
    let path = fixture_dir().join("open_clip_config.json");
    let raw = std::fs::read_to_string(&path).expect("read tiny_open_clip config");
    let json: serde_json::Value =
        serde_json::from_str(&raw).expect("parse tiny_open_clip config json");
    ClipTextConfig::from_open_clip_config(&json).expect("parse text config")
}

#[test]
fn clip_text_loads_from_fixture() {
    let device = Device::Cpu;
    let cfg = load_config();
    let weights = fixture_dir().join("open_clip_model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights.as_path()], DType::F32, &device)
            .expect("mmap open_clip_model.safetensors")
    };

    let model = ClipText::load(vb, &cfg).expect("construct ClipText from fixture");
    assert_eq!(model.embed_dim(), cfg.embed_dim);
    assert_eq!(model.context_length(), cfg.context_length);
}

#[test]
fn clip_text_forward_shape_and_projection_dim() {
    let device = Device::Cpu;
    let cfg = load_config();
    let weights = fixture_dir().join("open_clip_model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights.as_path()], DType::F32, &device).unwrap()
    };
    let model = ClipText::load(vb, &cfg).unwrap();

    // 4 sequences, length 6 each, with the EOT token (vocab_size-1) at the
    // last real position and padding zeros after.
    let eot = (cfg.vocab_size - 1) as u32;
    let ids: Vec<u32> = vec![
        1, 2, 3, 4, 5, eot, // EOT at index 5
        7, 8, 9, eot, 0, 0, // EOT at index 3
        10, 11, eot, 0, 0, 0, // EOT at index 2
        12, eot, 0, 0, 0, 0, // EOT at index 1
    ];
    let input_ids = Tensor::from_vec(ids, (4, 6), &device).unwrap();
    let mask = Tensor::ones((4, 6), DType::U32, &device).unwrap();

    let out = model.forward(&input_ids, &mask).expect("forward");
    // Output dim must be the projected embed_dim, NOT the per-token width.
    assert_eq!(out.dims(), &[4, cfg.embed_dim]);

    // Every row L2-normalized.
    let rows = out.to_vec2::<f32>().unwrap();
    for row in &rows {
        let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }
}
