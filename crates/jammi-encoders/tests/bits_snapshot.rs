//! K4: the three cross-modal towers' eval output at f32 is BYTE-IDENTICAL,
//! ON ONE BOX, to what they produced before LoRA sites, builders and
//! dtype-following masks existed.
//!
//! The hashes below are a snapshot of the BASE behaviour at `main@3a3010d5`
//! (the commit this unit branches from), on CPU at f32, over the exact
//! inputs this file reproduces. If a change here moves a single output bit
//! ON A PINNED BOX, one of them fails, and the only correct responses are
//! to fix the change or to declare the eval-bit change explicitly. They are
//! never to be "re-baselined" to whatever the code now produces — that
//! would convert this oracle into a tautology.
//!
//! # What this pin claims: a SAME-BOX pin
//!
//! A CPU-float bit literal in this file is a SAME-BOX oracle. It is not a
//! same-`(target_arch, target_os)` pin, and — measured the hard way — it is
//! not even a same-CPU-MODEL pin.
//!
//! That was forced by measurement. Keying the x86_64-Linux row by
//! `(arch, os)` was tried first and failed on the heterogeneous GitHub
//! runner fleet; the key was then narrowed to the runtime CPU MODEL STRING,
//! and THAT failed too. Five hermetic x86_64 runs of this test are on
//! record:
//!
//! - `clip_text` was identical on every run, on every host, x86 and arm.
//! - `open_clip_vision` was `15745465420605586933` on all five x86 runs.
//! - `htsat_audio` SPLIT: three runs (two scratch runs at base and tip, one
//!   PR run), all reporting CPU model `AMD EPYC 7763 64-Core Processor`,
//!   produced `6453805038327953572`; a post-merge run on main reporting the
//!   SAME model string produced `11635368092211268310`, as did an earlier
//!   run on an unidentified x86_64 runner.
//!
//! Those numbers are recorded here as the fleet-variance RECORD. They are
//! deliberately NOT asserted anywhere, and there is no x86 row in
//! `pinned_bits`: two different HTSAT values were observed under one
//! identical key, so no literal keyed that way can be a sound oracle.
//! Pasting either one back in would make the test a coin flip.
//!
//! Working HYPOTHESIS for the split — stated as a hypothesis, not a proven
//! cause, and proving it is not this crate's job: within one x86 CPU model
//! string, hosted VM slices differ in the cache topology they report, and
//! the `gemm` crate's cache-aware blocking then picks a different block
//! schedule, which changes the accumulation ORDER of HTSAT's larger
//! matmuls. Float addition is not associative, so a different fold order is
//! a different sum in the last bits. Consistent with that, HTSAT — the
//! deepest tower, with the largest matmuls — is the arm that moves while
//! the two small OpenCLIP towers do not; and on the pinning box (Apple M5
//! Pro) forcing `RAYON_NUM_THREADS` to each of 1, 2, 4 and 16 left all
//! three hashes unchanged, so this is not a thread-count effect that fixing
//! a thread count here could pin down. Whatever the cause, it lives in
//! `gemm`/candle and in the host, several layers below this crate, and no
//! diff in `jammi-encoders` can make a float sum associative across two
//! block schedules.
//!
//! So the surviving literal row answers exactly one question: "on THIS BOX,
//! did this crate's eval output bits move?".
//!
//! # How the cross-COMMIT half of K4 is actually proven
//!
//! Not here, and never by a literal carried across machines. "Base and tip
//! agree bit-for-bit" is only meaningful when BOTH are measured on the SAME
//! BOX, so it is a PROCEDURE, not a constant: the acceptance-verifier's
//! two-worktree run (base and tip built and run on one machine, each with
//! its own target dir) and the pod's pre-flight do exactly that. This file
//! cannot check it and does not pretend to; the literal row is the cheap
//! same-box regression tripwire that runs alongside it.
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
//! The genuinely host-independent checks are unconditional and run FIRST on
//! every host, pinned or not: the three output SHAPES, and the digest's own
//! non-vacuity (`fnv_digest_separates_a_single_flipped_bit`). Only the
//! literal comparison is box-scoped.
//!
//! On a host with no row the literal comparison CANNOT silently pass: the
//! model string, arch, OS and the three measured hashes are printed with
//! instructions to add a LEAD-COMPUTED row measured at the base commit on
//! that box, and `k4_snapshot_require_gate` turns the skip into a hard
//! failure for any lane that sets `JAMMI_REQUIRE_K4_SNAPSHOT`.
//!
//! That env var is for a KNOWN PINNED BOX — a machine with a row here, so
//! that a skip on it would be a defect. The GitHub-hosted fleet is not such
//! a box (see the record above), so `ci.yml` does NOT set it; a hosted lane
//! keeps the shape asserts and the non-vacuity control, and prints the
//! hashes when it lands on an unpinned host. A lane on genuinely fixed
//! hardware may set it once that box has a lead-computed row.
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

/// One BOX's row of lead-computed `main@3a3010d5` snapshots, CPU f32.
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
/// to `None` — an unmeasured host is unpinned, never approximately pinned.
///
/// There is deliberately NO x86_64 row: one CPU model string on the hosted
/// fleet produced two different HTSAT hashes (see the module doc), so
/// `(arch, os, cpu model)` does not identify a box there. The only rows
/// that belong here are boxes whose bits the lead measured directly, and
/// repeatedly, at the base commit.
fn pinned_bits(cpu_model: &str) -> Option<TowerBits> {
    match (std::env::consts::ARCH, std::env::consts::OS, cpu_model) {
        ("aarch64", "macos", "Apple M5 Pro") => Some(APPLE_M5_PRO),
        _ => None,
    }
}

/// Registered KO-7 require-gate helper (`ci/kernel-oracle-helpers.txt`) for
/// the no-pinned-row skip below. A lane that means to PROVE the K4 snapshot
/// (a lane on a known pinned box) sets `JAMMI_REQUIRE_K4_SNAPSHOT`; if that
/// lane ever runs on a host this file has no literals for, the missing pin
/// is a hard failure, never a silent green. CI does not set it — the
/// hosted fleet is not a pinned box.
fn k4_snapshot_require_gate() {
    if std::env::var_os("JAMMI_REQUIRE_K4_SNAPSHOT").is_some() {
        panic!(
            "tower_eval_output_bits_match_the_pre_lora_snapshot: \
             JAMMI_REQUIRE_K4_SNAPSHOT is set but this host has no row in \
             `pinned_bits` -- the K4 eval-bit snapshot cannot be proven here; \
             add a lead-computed row measured on THIS box, or run the \
             same-box base-vs-tip procedure, instead of accepting a skip"
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
    // Structural pin — host-independent, so it is asserted BEFORE any
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
            "K4 snapshot: UNPINNED HOST -- cpu model={cpu:?} arch={} os={} has no row in \
             `pinned_bits`. Measured on this host: \
             clip_text=0x{got_text:016x} ({got_text}) \
             open_clip_vision=0x{got_vision:016x} ({got_vision}) \
             htsat_audio=0x{got_audio:016x} ({got_audio}). \
             These are NOT verified against anything -- do not paste them in as a pin; \
             a CPU-float bit literal is a SAME-BOX oracle, and one hosted CPU model string \
             has already produced two different HTSAT hashes. Prove a cross-commit K4 claim \
             with a same-box base-vs-tip run instead; only add a row here for a box whose \
             bits the lead measured directly at the base commit.",
            std::env::consts::ARCH,
            std::env::consts::OS,
        );
        k4_snapshot_require_gate();
        return;
    };

    assert_eq!(
        got_text, pin.clip_text,
        "clip_text eval output bits changed vs main@3a3010d5 on this box ({cpu})"
    );
    assert_eq!(
        got_vision, pin.open_clip_vision,
        "open_clip_vision eval output bits changed vs main@3a3010d5 on this box ({cpu})"
    );
    assert_eq!(
        got_audio, pin.htsat_audio,
        "htsat_audio eval output bits changed vs main@3a3010d5 on this box ({cpu})"
    );
}

/// The digest is non-vacuous: it distinguishes tensors that differ in ONE
/// bit of ONE element. Without this control, a hash function that (say)
/// returned its seed on every input would make the three assertions above
/// pass for the wrong reason. Host-independent, so it runs everywhere,
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
