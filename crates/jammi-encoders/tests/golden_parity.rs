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
use jammi_encoders::htsat_audio::{HtsatAudio, HtsatAudioConfig, HtsatAudioEncoder};

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

    let front = encoder
        .forward_front(&input, &[true, true])
        .expect("front-half forward");

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

/// The `is_longer=false` patch-embed path: a short clip's patch embedding is the
/// global patch-conv ALONE, with no AFF fusion (HF `ClapAudioPatchEmbed` skips
/// `mel_conv2d`+`fusion_model` when `is_longer_idx` is empty). Driving the front
/// half with `is_longer=[false, false]` on the SAME pinned input must reproduce
/// the `patch_embed_out_global_only` golden — proving the per-sample fusion gate,
/// not the unconditional-fusion path that produces a different (wrong) embedding.
#[test]
fn patch_embed_global_only_matches_golden() {
    let device = Device::Cpu;
    let goldens = Goldens::load().expect("load goldens.safetensors");
    let encoder = load_front_encoder(&device);
    let input = load_pinned_input().expect("load pinned_input");

    let front = encoder
        .forward_front(&input, &[false, false])
        .expect("front-half forward (is_longer=false)");

    assert_close_max_abs(
        &front.patch_embed_out,
        goldens
            .get("patch_embed_out_global_only")
            .expect("patch_embed_out_global_only"),
        "patch_embed_out_global_only",
    );

    // The gate is real: the is_longer=true patch embed differs from the
    // global-only one on this input (the fusion path changes the embedding).
    let fused = encoder
        .forward_front(&input, &[true, true])
        .expect("front-half forward (is_longer=true)");
    let delta = max_abs_diff(&fused.patch_embed_out, &front.patch_embed_out)
        .expect("fused vs global-only diff");
    assert!(
        delta > TOL_ABS,
        "is_longer gate is a no-op: fused and global-only patch embeds agree (Δ={delta})"
    );
}

/// Build the full HTSAT-Swin tower (encoder + projection) from the committed
/// `model.safetensors` (root scope).
fn load_full_tower(device: &Device) -> HtsatAudio {
    let config = load_config();
    let weights = fixture_dir().join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)
            .expect("mmap model.safetensors")
    };
    HtsatAudio::load(vb, &config, device).expect("load full HTSAT tower")
}

/// Drive the full HTSAT-Swin tower on the pinned input and gate EVERY boundary
/// through the final L2-normalized embedding: the front half, every Swin block
/// and patch-merging downsample across all four stages, the final LayerNorm,
/// the group-2D pooling reshape and pool, and the projection head (unnormalized
/// max-abs + normalized cosine). A divergence localizes to its unit.
#[test]
fn full_forward_matches_goldens() {
    let device = Device::Cpu;
    let goldens = Goldens::load().expect("load goldens.safetensors");
    let tower = load_full_tower(&device);
    let input = load_pinned_input().expect("load pinned_input");

    // Front half (gated again here so the full tower stands alone).
    let front = tower
        .encoder()
        .forward_front(&input, &[true, true])
        .expect("front-half forward");
    let frames_num = front
        .post_reshape_mel2img
        .dim(2)
        .expect("frames_num from reshape_mel2img");

    let spine = tower
        .encoder()
        .forward_spine(&front.patch_embed_out, frames_num)
        .expect("spine forward");

    // Per-stage block + downsample boundaries.
    for (s, stage_blocks) in spine.blocks.iter().enumerate() {
        for (b, out) in stage_blocks.iter().enumerate() {
            let name = format!("stage{s}.block{b}_out");
            assert_close_max_abs(out, goldens.get(&name).expect("block golden"), &name);
        }
        if let Some(ds) = &spine.downsamples[s] {
            let name = format!("stage{s}.downsample_out");
            assert_close_max_abs(ds, goldens.get(&name).expect("downsample golden"), &name);
        }
    }

    assert_close_max_abs(
        &spine.final_norm_out,
        goldens.get("final_norm_out").expect("final_norm_out"),
        "final_norm_out",
    );
    assert_close_max_abs(
        &spine.pre_pool,
        goldens.get("pre_pool").expect("pre_pool"),
        "pre_pool",
    );
    assert_close_max_abs(
        &spine.pooler_out,
        goldens.get("pooler_out").expect("pooler_out"),
        "pooler_out",
    );

    // Projection head: unnormalized (max-abs) then normalized (cosine).
    let unnorm = tower
        .projection()
        .forward_unnormalized(&spine.pooler_out)
        .expect("projection forward");
    assert_close_max_abs(
        &unnorm,
        goldens
            .get("projected_unnormalized")
            .expect("projected_unnormalized"),
        "projected_unnormalized",
    );

    let normalized = tower.forward(&input, &[true, true]).expect("full forward");
    assert_close_cosine(
        &normalized,
        goldens
            .get("projected_normalized")
            .expect("projected_normalized"),
        "projected_normalized",
    );
}

/// The tower must accept ANY input time length, not the fixture's 500: the
/// bicubic resample builds its matrix from the runtime `T` and maps it to
/// `spec_width`, after which the geometry is config-driven and T-independent.
/// This is what lets the real `laion/clap-htsat-fused` (T≈1001) run; a tower
/// pinned to one length would silently work only on the tiny fixture. Drives a
/// length below `spec_width` (resampled) and one equal to it (resample skipped),
/// asserting a well-formed unit-norm embedding either way.
#[test]
fn tower_accepts_arbitrary_input_length() {
    let device = Device::Cpu;
    let tower = load_full_tower(&device);
    let n_mels = load_config().num_mel_bins;

    for t in [300usize, 512] {
        let input = Tensor::randn(0f32, 1f32, (2, 4, t, n_mels), &device).expect("random input");
        let emb = tower
            .forward(&input, &[true, true])
            .unwrap_or_else(|e| panic!("forward at T={t}: {e}"));
        assert_eq!(
            emb.dims(),
            &[2, tower.projection_dim()],
            "T={t}: embedding shape"
        );
        // L2-normalized rows: each row's norm ≈ 1 (and finite).
        let norms = emb.sqr().unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
        let min = norms.min(0).unwrap().to_scalar::<f32>().unwrap();
        let max = norms.max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(
            min.is_finite() && (min - 1.0).abs() < 1e-4 && (max - 1.0).abs() < 1e-4,
            "T={t}: embedding rows not unit-norm: [{min}, {max}]"
        );
    }
}

// ── Weight-key coverage ──────────────────────────────────────────────────────
//
// The boundary-parity tests above prove the tower is numerically correct on the
// committed input; this test proves it consumes the FULL checkpoint — every
// learned weight is loaded, with the right shape, and the tower invents no key
// the checkpoint lacks. It derives the expected `{key → shape}` set purely from
// the parsed `HtsatAudioConfig` (looping stages × blocks), then asserts an exact
// bijection with `model.safetensors`'s non-ignored keys: no checkpoint key is
// unexpected, no expected key is missing, every shape matches.
//
// Ignore-set invariant: exactly two key families carry no learned parameter and
// are intentionally NOT loaded by the tower —
//   * `*.relative_position_index` — the Swin window's pairwise index buffer,
//     recomputed from the window size in `SwinSelfAttention::build_rel_index`
//     (verified bit-exact against the stored buffer by the parity tests), so the
//     stored copy is redundant.
//   * `*.num_batches_tracked` — a BatchNorm step counter used only by training's
//     running-stat momentum; eval-mode inference reads `running_mean/var`, never
//     the counter.
// Both are non-parametric bookkeeping. Every OTHER key must be a real weight the
// tower loads; the ignore-set is closed (nothing else may be skipped).

/// Expected `{key → shape}` derived from config: append the `audio_encoder`
/// prefix and collect.
fn expected_weight_shapes(config: &HtsatAudioConfig) -> HashMap<String, Vec<usize>> {
    let mut want: HashMap<String, Vec<usize>> = HashMap::new();
    let enc = "audio_model.audio_encoder";
    let mut put = |key: String, shape: Vec<usize>| {
        want.insert(key, shape);
    };

    // Input batch-norm over the mel axis: weight/bias/running_mean/running_var.
    for p in ["weight", "bias", "running_mean", "running_var"] {
        put(format!("{enc}.batch_norm.{p}"), vec![config.num_mel_bins]);
    }

    // Patch embedding.
    let phs = config.patch_embeds_hidden_size;
    let in_c = config.patch_embed_input_channels;
    let ps = config.patch_size;
    let pe = format!("{enc}.patch_embed");
    put(format!("{pe}.proj.weight"), vec![phs, in_c, ps, ps]);
    put(format!("{pe}.proj.bias"), vec![phs]);
    put(
        format!("{pe}.mel_conv2d.weight"),
        vec![phs, in_c, ps, ps * 3],
    );
    put(format!("{pe}.mel_conv2d.bias"), vec![phs]);
    put(format!("{pe}.norm.weight"), vec![phs]);
    put(format!("{pe}.norm.bias"), vec![phs]);

    // AFF fusion block: two MLP branches (local/global), each conv → BN → conv →
    // BN with a `phs/aff_block_r` bottleneck. The Sequential indices match the
    // HF layout: convs at 0/3 (local) and 1/4 (global), BNs at 1/4 and 2/5.
    let inter = phs / config.aff_block_r;
    let conv = |w: &mut dyn FnMut(String, Vec<usize>), base: &str, out: usize, inp: usize| {
        w(format!("{base}.weight"), vec![out, inp, 1, 1]);
        w(format!("{base}.bias"), vec![out]);
    };
    let bn = |w: &mut dyn FnMut(String, Vec<usize>), base: &str, c: usize| {
        for p in ["weight", "bias", "running_mean", "running_var"] {
            w(format!("{base}.{p}"), vec![c]);
        }
    };
    let fm = format!("{pe}.fusion_model");
    conv(&mut put, &format!("{fm}.local_att.0"), inter, phs);
    bn(&mut put, &format!("{fm}.local_att.1"), inter);
    conv(&mut put, &format!("{fm}.local_att.3"), phs, inter);
    bn(&mut put, &format!("{fm}.local_att.4"), phs);
    conv(&mut put, &format!("{fm}.global_att.1"), inter, phs);
    bn(&mut put, &format!("{fm}.global_att.2"), inter);
    conv(&mut put, &format!("{fm}.global_att.4"), phs, inter);
    bn(&mut put, &format!("{fm}.global_att.5"), phs);

    // Swin stages: `depths[s]` blocks at width `phs << s`, then a patch-merging
    // downsample on every stage but the last.
    let num_stages = config.num_stages();
    for s in 0..num_stages {
        let dim = phs << s;
        let inter_mlp = (config.mlp_ratio * dim as f64) as usize;
        let heads = config.num_attention_heads[s];
        for b in 0..config.depths[s] {
            let blk = format!("{enc}.layers.{s}.blocks.{b}");
            // Pre/post-attention LayerNorms.
            for ln in ["layernorm_before", "layernorm_after"] {
                put(format!("{blk}.{ln}.weight"), vec![dim]);
                put(format!("{blk}.{ln}.bias"), vec![dim]);
            }
            // Window self-attention: q/k/v + the relative-position bias table.
            for proj in ["query", "key", "value"] {
                put(
                    format!("{blk}.attention.self.{proj}.weight"),
                    vec![dim, dim],
                );
                put(format!("{blk}.attention.self.{proj}.bias"), vec![dim]);
            }
            // HF sizes the bias table by the config window: (2·ws−1)² rows.
            let ws = config.window_size;
            let table_rows = (2 * ws - 1) * (2 * ws - 1);
            put(
                format!("{blk}.attention.self.relative_position_bias_table"),
                vec![table_rows, heads],
            );
            // Attention output projection.
            put(
                format!("{blk}.attention.output.dense.weight"),
                vec![dim, dim],
            );
            put(format!("{blk}.attention.output.dense.bias"), vec![dim]);
            // MLP: intermediate (dim → inter) then output (inter → dim).
            put(
                format!("{blk}.intermediate.dense.weight"),
                vec![inter_mlp, dim],
            );
            put(format!("{blk}.intermediate.dense.bias"), vec![inter_mlp]);
            put(format!("{blk}.output.dense.weight"), vec![dim, inter_mlp]);
            put(format!("{blk}.output.dense.bias"), vec![dim]);
        }
        if s < num_stages - 1 {
            let ds = format!("{enc}.layers.{s}.downsample");
            put(format!("{ds}.norm.weight"), vec![4 * dim]);
            put(format!("{ds}.norm.bias"), vec![4 * dim]);
            put(format!("{ds}.reduction.weight"), vec![2 * dim, 4 * dim]);
        }
    }

    // Final encoder LayerNorm at the widest width.
    put(format!("{enc}.norm.weight"), vec![config.hidden_size]);
    put(format!("{enc}.norm.bias"), vec![config.hidden_size]);

    // Projection head (sibling of `audio_model`): linear1 → act → linear2.
    let pd = config.projection_dim;
    put(
        "audio_projection.linear1.weight".into(),
        vec![pd, config.hidden_size],
    );
    put("audio_projection.linear1.bias".into(), vec![pd]);
    put("audio_projection.linear2.weight".into(), vec![pd, pd]);
    put("audio_projection.linear2.bias".into(), vec![pd]);

    want
}

/// A checkpoint key carries no learned parameter (recomputed index buffer or BN
/// step counter) and is therefore intentionally not loaded by the tower.
fn is_ignored_key(key: &str) -> bool {
    key.ends_with("relative_position_index") || key.ends_with("num_batches_tracked")
}

#[test]
fn every_checkpoint_weight_is_loaded_with_the_right_shape() {
    let config = load_config();
    let weights = fixture_dir().join("model.safetensors");
    let checkpoint =
        candle_core::safetensors::load(&weights, &Device::Cpu).expect("load model.safetensors");

    let expected = expected_weight_shapes(&config);

    // 1. Every non-ignored checkpoint key is expected, with the right shape.
    let mut extras = Vec::new();
    let mut shape_mismatches = Vec::new();
    for (key, tensor) in &checkpoint {
        if is_ignored_key(key) {
            continue;
        }
        match expected.get(key) {
            None => extras.push(key.clone()),
            Some(want_shape) => {
                if tensor.dims() != want_shape.as_slice() {
                    shape_mismatches.push(format!(
                        "{key}: checkpoint {:?} != expected {:?}",
                        tensor.dims(),
                        want_shape
                    ));
                }
            }
        }
    }

    // 2. Every expected key is present in the checkpoint.
    let mut missing: Vec<String> = expected
        .keys()
        .filter(|k| !checkpoint.contains_key(*k))
        .cloned()
        .collect();

    extras.sort();
    missing.sort();
    shape_mismatches.sort();

    assert!(
        extras.is_empty(),
        "checkpoint keys not derived from config (the tower would silently ignore real \
         weights, or the ignore-set is too narrow): {extras:#?}"
    );
    assert!(
        missing.is_empty(),
        "config-derived keys absent from the checkpoint (the tower expects weights that \
         don't exist): {missing:#?}"
    );
    assert!(
        shape_mismatches.is_empty(),
        "shape mismatches between checkpoint and config-derived expectation: {shape_mismatches:#?}"
    );

    // The ignore-set is real, not a way to dodge coverage: every ignored key
    // must be one of the two non-parametric families, and the non-ignored count
    // must exactly equal the derived expectation.
    let non_ignored = checkpoint.keys().filter(|k| !is_ignored_key(k)).count();
    assert_eq!(
        non_ignored,
        expected.len(),
        "non-ignored checkpoint key count must equal the config-derived expectation"
    );
}
