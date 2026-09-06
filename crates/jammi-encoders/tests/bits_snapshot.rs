//! K4: the three cross-modal towers' eval output at f32 is BYTE-IDENTICAL,
//! ON ONE PLATFORM, to what they produced before LoRA sites, builders and
//! dtype-following masks existed.
//!
//! The hashes below are a snapshot of the BASE behaviour at `main@3a3010d5`
//! (the commit this unit branches from), on CPU at f32, over the exact
//! inputs this file reproduces. If a change here moves a single output bit,
//! one of them fails, and the only correct responses are to fix the change
//! or to declare the eval-bit change explicitly. They are never to be
//! "re-baselined" to whatever the code now produces — that would convert
//! this oracle into a tautology.
//!
//! # What this pin does and does not claim
//!
//! It is a SAME-PLATFORM pin. Cross-platform bit equality is NOT claimed and
//! is not true: the vision tower's output differs between aarch64-macOS and
//! x86_64-Linux, because the gemm/conv dispatch chosen for its shapes differs
//! per platform. That was measured, not assumed — the x86_64-Linux vision
//! value here was computed on the CI image at BASE (`main@3a3010d5`), where
//! no LoRA site exists at all, so the divergence is a property of the host's
//! float reduction order, not of this unit's diff. The text and audio towers
//! happen to agree across the two pinned platforms; nothing guarantees that
//! for a third.
//!
//! Consequently the literals live in a per-platform table
//! (`pinned_bits`), and the checks that are genuinely platform-independent —
//! output SHAPES, and the digest's own non-vacuity — run everywhere,
//! unconditionally. The other structural pin, `builder().lora(frozen())`
//! being bit-identical to `load()`, is an EQUALITY BETWEEN TWO TENSORS
//! COMPUTED ON THE SAME HOST, so it needs no literal at all and is asserted
//! on every platform by `tower_lora.rs`'s
//! `a2_builder_frozen_and_zerosb_adapter_are_bit_identical_to_load`; it is
//! deliberately not duplicated here.
//!
//! On a platform with no row, the literal comparison CANNOT silently pass:
//! the three hashes are printed with instructions, and
//! `k4_snapshot_require_gate` turns the skip into a hard failure for any lane
//! that sets `JAMMI_REQUIRE_K4_SNAPSHOT`.
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

/// One platform's row of lead-computed `main@3a3010d5` snapshots, CPU f32.
struct TowerBits {
    clip_text: u64,
    open_clip_vision: u64,
    htsat_audio: u64,
}

/// Computed by the lead on Apple-silicon macOS (aarch64), CPU f32.
const AARCH64_MACOS: TowerBits = TowerBits {
    clip_text: 0x6345_1755_4de7_ed36,
    open_clip_vision: 0xaee7_cdd1_c4f5_6e01,
    htsat_audio: 0x5990_857d_1729_f8a4,
};

/// Computed on the hermetic CI image (`ghcr.io/f-inverse/jammi-ai-ci`) on an
/// AMD EPYC 7763 — Zen 3, so AVX2 and no AVX512 — CPU f32, at BASE
/// (`main@3a3010d5`). The vision value `0xda83_2160_8264_7df5` is
/// `15745465420605586933` decimal, the number that job actually reported.
///
/// CAVEAT: this row is only as stable as the gemm crate's RUNTIME ISA
/// dispatch. It is pinned against a Zen-3 host; an x86_64 box that offers
/// AVX512 may select different kernels and hash differently, and nothing
/// here pins the ISA. If an x86_64 lane on other hardware fails only the
/// vision arm, that is the likely cause — split this row by ISA rather than
/// re-baselining it.
const X86_64_LINUX: TowerBits = TowerBits {
    clip_text: 0x6345_1755_4de7_ed36,
    open_clip_vision: 0xda83_2160_8264_7df5,
    htsat_audio: 0x5990_857d_1729_f8a4,
};

/// The row for THIS build target, or `None` when no lead-computed row exists.
fn pinned_bits() -> Option<TowerBits> {
    if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
        Some(AARCH64_MACOS)
    } else if cfg!(target_arch = "x86_64") && cfg!(target_os = "linux") {
        Some(X86_64_LINUX)
    } else {
        None
    }
}

/// Registered KO-7 require-gate helper (`ci/kernel-oracle-helpers.txt`) for
/// the no-pinned-row skip below. A lane that means to PROVE the K4 snapshot
/// (the hermetic job) sets `JAMMI_REQUIRE_K4_SNAPSHOT`; if that lane ever
/// runs on a platform this file has no literals for, the missing pin is a
/// hard failure, never a silent green.
fn k4_snapshot_require_gate() {
    if std::env::var_os("JAMMI_REQUIRE_K4_SNAPSHOT").is_some() {
        panic!(
            "tower_eval_output_bits_match_the_pre_lora_snapshot: \
             JAMMI_REQUIRE_K4_SNAPSHOT is set but this target has no row in \
             `pinned_bits` -- the K4 eval-bit snapshot cannot be proven here; \
             add a lead-computed row for this (target_arch, target_os) instead \
             of accepting a skip"
        );
    }
}

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
    // Structural pin — platform-independent, so it is asserted BEFORE any
    // possible skip below.
    assert_eq!(out_t.dims(), &[2, 16]);
    let got_text = fnv(&out_t);

    let vcfg = OpenClipVisionConfig::from_open_clip_config(&cfg).unwrap();
    let vision = OpenClipVisionTransformer::load(vb.pp("visual"), &vcfg).unwrap();
    let n = 2 * 3 * 8 * 8;
    let px: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.017 - 1.0).sin() * 0.5)
        .collect();
    let pixel = Tensor::from_vec(px, (2, 3, 8, 8), &dev).unwrap();
    let out_v = vision.forward(&pixel).unwrap();
    assert_eq!(out_v.dims(), &[2, 16]);
    let got_vision = fnv(&out_v);

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
    let got_audio = fnv(&out_a);

    let Some(pin) = pinned_bits() else {
        println!(
            "K4 snapshot: UNPINNED PLATFORM (target_arch/target_os has no row in \
             `pinned_bits`) -- add a lead-computed row for it. Measured on this \
             host: clip_text=0x{got_text:016x} open_clip_vision=0x{got_vision:016x} \
             ({got_vision}) htsat_audio=0x{got_audio:016x}. These are NOT verified \
             against anything; do not paste them in without a lead measurement."
        );
        k4_snapshot_require_gate();
        return;
    };

    assert_eq!(
        got_text, pin.clip_text,
        "clip_text eval output bits changed vs main@3a3010d5 on this platform"
    );
    assert_eq!(
        got_vision, pin.open_clip_vision,
        "open_clip_vision eval output bits changed vs main@3a3010d5 on this platform"
    );
    assert_eq!(
        got_audio, pin.htsat_audio,
        "htsat_audio eval output bits changed vs main@3a3010d5 on this platform"
    );
}

/// The digest is non-vacuous: it distinguishes tensors that differ in ONE
/// bit of ONE element. Without this control, a hash function that (say)
/// returned its seed on every input would make the three assertions above
/// pass for the wrong reason. Platform-independent, so it runs everywhere,
/// including on a platform with no pinned row.
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
