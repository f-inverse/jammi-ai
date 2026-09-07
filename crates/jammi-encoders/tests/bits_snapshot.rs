//! K4: the three cross-modal towers' eval output at f32 is BYTE-IDENTICAL,
//! ON ONE CPU MODEL, to what they produced before LoRA sites, builders and
//! dtype-following masks existed.
//!
//! The hashes below are a snapshot of the BASE behaviour at `main@3a3010d5`
//! (the commit this unit branches from), on CPU at f32, over the exact
//! inputs this file reproduces. If a change here moves a single output bit
//! ON A PINNED CPU MODEL, one of them fails, and the only correct responses
//! are to fix the change or to declare the eval-bit change explicitly. They
//! are never to be "re-baselined" to whatever the code now produces — that
//! would convert this oracle into a tautology.
//!
//! # What this pin claims, and the unit it is claimed in
//!
//! It is a SAME-CPU-MODEL pin, not a same-`(target_arch, target_os)` pin.
//! The literal table `pinned_bits` is therefore keyed by the runtime CPU
//! MODEL STRING (`cpu_model`), not by build target alone.
//!
//! That key was forced by measurement, not chosen for caution. Keying the
//! x86_64-Linux row by `(arch, os)` was tried first and failed on the
//! GitHub runner fleet, which is heterogeneous: three hermetic CI runs that
//! landed on an `AMD EPYC 7763 64-Core Processor` agreed exactly (vision
//! `15745465420605586933`, audio `0x5990_857d_1729_f8a4`), while a fourth
//! run of the SAME COMMIT on a different, unidentified x86_64 runner
//! reported audio `11635368092211268310` — the same arch and OS, a
//! different number. On the pinning box (Apple M5 Pro, 18 cores) forcing
//! `RAYON_NUM_THREADS` to each of 1, 2, 4 and 16 left all three hashes
//! unchanged, so the divergence is NOT a thread-count-dependent reduction
//! order: it is gemm/conv kernel dispatch selected from the CPU model's
//! ISA. That dispatch lives in candle/gemm, several layers below this
//! crate; it is a property of those crates and of the host, and no diff in
//! `jammi-encoders` can make a float sum associative across two different
//! kernel selections.
//!
//! So the literals here answer exactly one question: "on THIS CPU model,
//! did this crate's eval output bits move?". The cross-COMMIT half of the
//! K4 claim — that base and tip agree — is only meaningful when both are
//! measured on the SAME BOX, which is the acceptance-verifier's job (it
//! builds base and tip on one machine) and the pod's, not this test's.
//! This file cannot check it, and does not pretend to.
//!
//! The other structural pin, `builder().lora(frozen())` being bit-identical
//! to `load()`, is an EQUALITY BETWEEN TWO TENSORS COMPUTED ON THE SAME
//! HOST in the same process, so it needs no literal at all and holds on
//! every CPU; it is asserted by `tower_lora.rs`'s
//! `a2_builder_frozen_and_zerosb_adapter_are_bit_identical_to_load` and
//! deliberately not duplicated here.
//!
//! # What runs everywhere
//!
//! The genuinely CPU-independent checks are unconditional and run FIRST on
//! every host, pinned or not: the three output SHAPES, and the digest's own
//! non-vacuity (`fnv_digest_separates_a_single_flipped_bit`). Only the
//! literal comparison is CPU-model-scoped.
//!
//! On a CPU model with no row the literal comparison CANNOT silently pass:
//! the model string, arch, OS and the three measured hashes are printed
//! with instructions to add a LEAD-COMPUTED row measured at the base
//! commit on that CPU, and `k4_snapshot_require_gate` turns the skip into a
//! hard failure for any lane that sets `JAMMI_REQUIRE_K4_SNAPSHOT`.
//!
//! That env var is for a KNOWN PINNED BOX — a machine whose CPU model has a
//! row here, so that a skip on it would be a defect. The GitHub-hosted
//! fleet is not such a box (see the four-run measurement above), so
//! `ci.yml` does NOT set it; a hosted lane keeps the shape asserts and the
//! non-vacuity control, and prints the hashes when it lands on an unpinned
//! CPU. A self-hosted or containerised lane with a fixed CPU model may set
//! it once that model has a lead-computed row.
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

/// One CPU model's row of lead-computed `main@3a3010d5` snapshots, CPU f32.
struct TowerBits {
    clip_text: u64,
    open_clip_vision: u64,
    htsat_audio: u64,
}

/// Computed by the lead on an Apple M5 Pro (aarch64 macOS), CPU f32, at
/// BASE (`main@3a3010d5`).
const APPLE_M5_PRO: TowerBits = TowerBits {
    clip_text: 0x6345_1755_4de7_ed36,
    open_clip_vision: 0xaee7_cdd1_c4f5_6e01,
    htsat_audio: 0x5990_857d_1729_f8a4,
};

/// Computed on the hermetic CI image (`ghcr.io/f-inverse/jammi-ai-ci`),
/// x86_64 Linux, on an `AMD EPYC 7763 64-Core Processor` (Zen 3 — AVX2, no
/// AVX512), CPU f32, at BASE (`main@3a3010d5`). The vision value
/// `0xda83_2160_8264_7df5` is `15745465420605586933` decimal, the number
/// those jobs actually reported. Three separate runs on this CPU model
/// agreed on all three hashes; a run on a DIFFERENT x86_64 runner did not,
/// which is why the key is the model string and not `("x86_64", "linux")`.
const AMD_EPYC_7763: TowerBits = TowerBits {
    clip_text: 0x6345_1755_4de7_ed36,
    open_clip_vision: 0xda83_2160_8264_7df5,
    htsat_audio: 0x5990_857d_1729_f8a4,
};

/// This host's CPU model string, or `None` when it cannot be read.
///
/// Linux: the first `model name` value in `/proc/cpuinfo`. macOS:
/// `sysctl -n machdep.cpu.brand_string`. Both are shelled/read at runtime
/// rather than pulled from a crate — this is a test, and adding a
/// dependency to key a literal table would be the wrong trade.
fn detect_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in text.lines() {
            // A `?` here would be a bug: `/proc/cpuinfo` has blank and
            // colon-less lines, and bailing on the first one would report
            // "unreadable" on every Linux host.
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            if key.trim() == "model name" {
                let value = value.trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let value = String::from_utf8(out.stdout).ok()?;
        let value = value.trim();
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// `detect_cpu_model` with an explicit placeholder for "unreadable", so the
/// unpinned message always names something and an unreadable model can
/// never accidentally key a row.
fn cpu_model() -> String {
    detect_cpu_model().unwrap_or_else(|| "<unreadable>".to_string())
}

/// The row for this host, or `None` when no lead-computed row exists for
/// its `(arch, os, cpu model)`. A near-miss on the model string (a
/// different EPYC part, a different Apple part) intentionally falls through
/// to `None` — an unmeasured CPU is unpinned, never approximately pinned.
fn pinned_bits(cpu_model: &str) -> Option<TowerBits> {
    match (std::env::consts::ARCH, std::env::consts::OS, cpu_model) {
        ("aarch64", "macos", "Apple M5 Pro") => Some(APPLE_M5_PRO),
        ("x86_64", "linux", "AMD EPYC 7763 64-Core Processor") => Some(AMD_EPYC_7763),
        _ => None,
    }
}

/// Registered KO-7 require-gate helper (`ci/kernel-oracle-helpers.txt`) for
/// the no-pinned-row skip below. A lane that means to PROVE the K4 snapshot
/// (a lane on a known pinned box) sets `JAMMI_REQUIRE_K4_SNAPSHOT`; if that
/// lane ever runs on a CPU model this file has no literals for, the missing
/// pin is a hard failure, never a silent green.
fn k4_snapshot_require_gate() {
    if std::env::var_os("JAMMI_REQUIRE_K4_SNAPSHOT").is_some() {
        panic!(
            "tower_eval_output_bits_match_the_pre_lora_snapshot: \
             JAMMI_REQUIRE_K4_SNAPSHOT is set but this host has no row in \
             `pinned_bits` -- the K4 eval-bit snapshot cannot be proven here; \
             add a lead-computed row for this (arch, os, cpu model) instead \
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
    // Structural pin — CPU-independent, so it is asserted BEFORE any
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

    let cpu = cpu_model();
    let Some(pin) = pinned_bits(&cpu) else {
        println!(
            "K4 snapshot: UNPINNED CPU MODEL -- cpu model={cpu:?} arch={} os={} has no row in \
             `pinned_bits`. Measured on this host: \
             clip_text=0x{got_text:016x} ({got_text}) \
             open_clip_vision=0x{got_vision:016x} ({got_vision}) \
             htsat_audio=0x{got_audio:016x} ({got_audio}). \
             These are NOT verified against anything -- do not paste them in as a pin; \
             add a lead-computed row measured at the base commit on this CPU.",
            std::env::consts::ARCH,
            std::env::consts::OS,
        );
        k4_snapshot_require_gate();
        return;
    };

    assert_eq!(
        got_text, pin.clip_text,
        "clip_text eval output bits changed vs main@3a3010d5 on this CPU model ({cpu})"
    );
    assert_eq!(
        got_vision, pin.open_clip_vision,
        "open_clip_vision eval output bits changed vs main@3a3010d5 on this CPU model ({cpu})"
    );
    assert_eq!(
        got_audio, pin.htsat_audio,
        "htsat_audio eval output bits changed vs main@3a3010d5 on this CPU model ({cpu})"
    );
}

/// The digest is non-vacuous: it distinguishes tensors that differ in ONE
/// bit of ONE element. Without this control, a hash function that (say)
/// returned its seed on every input would make the three assertions above
/// pass for the wrong reason. CPU-independent, so it runs everywhere,
/// including on a host with no pinned row.
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
