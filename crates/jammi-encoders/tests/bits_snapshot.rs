//! K4: the three cross-modal towers' eval output at f32 is BYTE-IDENTICAL
//! to what they produced before LoRA sites, builders and dtype-following
//! masks existed.
//!
//! The three hashes below were computed by the lead on `main@3a3010d5` (the
//! commit this unit branches from), on CPU at f32, over the exact inputs
//! this file reproduces. They are a snapshot of the BASE behaviour, not of
//! this commit's: if a change here moves a single output bit, one of them
//! fails, and the only correct responses are to fix the change or to
//! declare the eval-bit change explicitly. They are never to be
//! "re-baselined" to whatever the code now produces — that would convert
//! this oracle into a tautology.
//!
//! FNV-1a over the little-endian bytes of every output `f32`'s bit pattern:
//! a bit-level digest, so a `NaN` payload difference or a sign-of-zero flip
//! shows up exactly like any other divergence (a value-level `==` would let
//! `NaN != NaN` pass silently, and `-0.0 == 0.0` would hide a sign flip).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use jammi_encoders::{
    ClipText, ClipTextConfig, HtsatAudio, HtsatAudioConfig, OpenClipVisionConfig,
    OpenClipVisionTransformer,
};

/// The lead's `main@3a3010d5` snapshot, CPU f32.
const CLIP_TEXT_BITS: u64 = 0x6345_1755_4de7_ed36;
/// See [`CLIP_TEXT_BITS`].
const OPEN_CLIP_VISION_BITS: u64 = 0xaee7_cdd1_c4f5_6e01;
/// See [`CLIP_TEXT_BITS`].
const HTSAT_AUDIO_BITS: u64 = 0x5990_857d_1729_f8a4;

fn fnv(t: &Tensor) -> u64 {
    let v: Vec<f32> = t
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn tower_eval_output_bits_match_the_pre_lora_snapshot() {
    let dev = Device::Cpu;

    // OpenCLIP tiny: text + vision, from the one committed checkpoint.
    let d = root().join("tests/fixtures/tiny_open_clip");
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("open_clip_config.json")).unwrap())
            .unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[d.join("open_clip_model.safetensors")],
            DType::F32,
            &dev,
        )
        .unwrap()
    };

    let tcfg = ClipTextConfig::from_open_clip_config(&cfg).unwrap();
    let text = ClipText::load(vb.clone(), &tcfg).unwrap();
    let ids: Vec<u32> = vec![1, 5, 9, 13, 96, 0, 0, 0, 1, 2, 3, 4, 5, 6, 96, 0];
    let input_ids = Tensor::from_vec(ids, (2, 8), &dev).unwrap();
    let mask = Tensor::ones((2, 8), DType::U32, &dev).unwrap();
    let out_t = text.forward(&input_ids, &mask).unwrap();
    assert_eq!(out_t.dims(), &[2, 16]);
    assert_eq!(
        fnv(&out_t),
        CLIP_TEXT_BITS,
        "clip_text eval output bits changed vs main@3a3010d5"
    );

    let vcfg = OpenClipVisionConfig::from_open_clip_config(&cfg).unwrap();
    let vision = OpenClipVisionTransformer::load(vb.pp("visual"), &vcfg).unwrap();
    let n = 2 * 3 * 8 * 8;
    let px: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017 - 1.0).sin() * 0.5)
        .collect();
    let pixel = Tensor::from_vec(px, (2, 3, 8, 8), &dev).unwrap();
    let out_v = vision.forward(&pixel).unwrap();
    assert_eq!(out_v.dims(), &[2, 16]);
    assert_eq!(
        fnv(&out_v),
        OPEN_CLIP_VISION_BITS,
        "open_clip_vision eval output bits changed vs main@3a3010d5"
    );

    // HTSAT tiny on its pinned input.
    let d = root().join("cookbook/fixtures/htsat_clap_tiny");
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    let acfg = HtsatAudioConfig::from_hf_clap_config(&cfg).unwrap();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[d.join("model.safetensors")], DType::F32, &dev)
            .unwrap()
    };
    let audio = HtsatAudio::load(vb, &acfg, &dev).unwrap();
    let pinned = candle_core::safetensors::load(d.join("pinned_input.safetensors"), &dev).unwrap();
    let feats = pinned.get("input_features").unwrap();
    let out_a = audio.forward(feats, &[true, true]).unwrap();
    assert_eq!(out_a.dims(), &[2, 8]);
    assert_eq!(
        fnv(&out_a),
        HTSAT_AUDIO_BITS,
        "htsat_audio eval output bits changed vs main@3a3010d5"
    );
}

/// The digest is non-vacuous: it distinguishes tensors that differ in ONE
/// bit of ONE element. Without this control, a hash function that (say)
/// returned its seed on every input would make the three assertions above
/// pass for the wrong reason.
#[test]
fn fnv_digest_separates_a_single_flipped_bit() {
    let dev = Device::Cpu;
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &dev).unwrap();
    let nudged = f32::from_bits(3.0f32.to_bits() + 1);
    let b = Tensor::from_vec(vec![1.0f32, 2.0, nudged], 3, &dev).unwrap();
    assert_ne!(fnv(&a), fnv(&b));
    // And a sign-of-zero flip, which a value-level `==` would not catch.
    let z = Tensor::from_vec(vec![0.0f32], 1, &dev).unwrap();
    let nz = Tensor::from_vec(vec![-0.0f32], 1, &dev).unwrap();
    assert_ne!(fnv(&z), fnv(&nz));
}
