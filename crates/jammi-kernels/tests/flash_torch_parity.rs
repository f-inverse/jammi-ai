//! O0(a) (P6 Stage B contract §4): parity vs torch's OWN vendored FA2
//! (`torch.ops.aten._flash_attention_forward`/`_backward`) on the B0
//! fixtures (`tests/fixtures/flash_reference/`, tracked, generator +
//! sidecar in the same directory — see that sidecar's
//! `version_mismatch_note` for the reference's exact identity: torch pins
//! `Dao-AILab/flash-attention` at commit `6c4f74fb338e` = release **2.8.4**
//! (`flash_attn/__init__.py:6`), the SAME `csrc/flash_attn/src` kernel
//! subtree jammi vendors at `v2.8.3.post1` (one patch release apart), but
//! torch's build defines `UNFUSE_FMA` and skips `--use_fast_math` while
//! jammi's does not — a cross-build (FMA-fusion / fast-math) reference,
//! not a same-flags one.
//!
//! Runs `crate::flash::flash_varlen_fwd`/`flash_varlen_bwd` DIRECTLY (the
//! FFI-boundary layer), not `ops::flash_attention_varlen` — this is where
//! `lse` is a first-class return value (the `ops` wrapper stashes it in a
//! private `Saved` slot, see that module's own doc for why); the `ops`
//! layer's own oracle suite (`tests/flash_op_oracles.rs`) covers the
//! `Tensor`/autograd wiring on top of this, not the numerics again.
//!
//! # Tolerance derivation (analytic, not a same-build self-diff)
//!
//! `o`/`lse`: forward has no split-KV path reachable from either build's
//! varlen entry (`crate::flash`'s own determinism doc), so two runs of the
//! SAME build are bit-identical — a same-build self-diff would read 0 and
//! prove nothing about a CROSS-build comparison. The bound used here is
//! `atol = k * 2^-8 * accumulation_depth * max(|reference|, 1)` where
//! `2^-8` is bf16's mantissa ULP (8 bits) and `accumulation_depth =
//! ceil(max_seqlen / 128)` (the number of `kBlockN=128` KV tiles the
//! online-softmax reduces over — `crate::flash`'s own module doc cites the
//! same tiling), `k = 8` as a deliberately loose safety factor covering
//! the UNFUSE_FMA/fast-math disagreement between builds (NVIDIA's
//! documented fast-math `exp2f` relative error is ~2^-11, two to three
//! orders below this bound — this crate's own design study §14/ADDENDUM
//! cites the same figure). `dq`/`dk`/`dv`: derived from the SAME formula
//! PLUS the sidecar's own measured same-build backward self-diff (both
//! builds' default backward accumulation), whichever is larger.
//!
//! # Injection control
//!
//! A deliberate `softmax_scale * 2` on the JAMMI side (reference fixture
//! generated at the correct scale) must RED this oracle — proves the
//! comparison is discriminating, not vacuously wide.

#![cfg(feature = "flash-attn")]

use std::path::{Path, PathBuf};

use candle_core::{CudaDevice, DType, Device, Tensor};
use half::bf16;
use jammi_kernels::flash::{CuSeqlens, VarlenConfig};

fn cuda_device() -> Option<CudaDevice> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d.as_cuda_device().unwrap().clone()),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "flash_torch_parity: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                     acquired — a silent skip here is not acceptable: {e}"
                );
            }
            eprintln!("flash_torch_parity: skipping — no CUDA device available ({e})");
            None
        }
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flash_reference")
}

fn load_f32(leg: &str, name: &str) -> Tensor {
    let path = fixtures_dir().join(format!("{leg}_{name}.npy"));
    Tensor::read_npy(&path).unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()))
}

/// One B0 leg: `(name, lengths, window_radius)`. Mirrors
/// `generate_fixtures.py`'s `LEGS` list exactly (H=4, D=64,
/// softmax_scale=1/8 for every leg — see that script and the sidecar).
struct Leg {
    name: &'static str,
    lengths: &'static [usize],
    window: Option<u32>,
}

const LEGS: &[Leg] = &[
    Leg {
        name: "b1_s512",
        lengths: &[512],
        window: None,
    },
    Leg {
        name: "b1_s512_win64",
        lengths: &[512],
        window: Some(64),
    },
    Leg {
        name: "b8_s512",
        lengths: &[512, 500, 512, 480, 512, 505, 490, 512],
        window: None,
    },
    Leg {
        name: "b8_s512_win64",
        lengths: &[512, 500, 512, 480, 512, 505, 490, 512],
        window: Some(64),
    },
    Leg {
        name: "b8_s128",
        lengths: &[128, 120, 128, 100, 128, 115, 128, 128],
        window: None,
    },
    Leg {
        name: "b1_tile129",
        lengths: &[129],
        window: None,
    },
    Leg {
        name: "b1_tile257",
        lengths: &[257],
        window: None,
    },
    Leg {
        name: "prefix_mixed",
        lengths: &[512, 300, 129, 64],
        window: None,
    },
];

const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const SOFTMAX_SCALE: f32 = 0.125; // 1/sqrt(64)

/// bf16 ULP-and-depth-derived tolerance — see the module doc's "Tolerance
/// derivation" section.
fn analytic_tolerance(max_seqlen: usize, reference_max_abs: f64) -> f64 {
    let depth = (max_seqlen as f64 / 128.0).ceil().max(1.0);
    let bf16_ulp = 2f64.powi(-8);
    let k = 8.0;
    k * bf16_ulp * depth * reference_max_abs.max(1.0)
}

/// Packs `q`/`k`/`v` ([total_q, H, D] each) into `[total_q, 3, H, D]` bf16
/// — the layout `crate::flash`'s FFI requires (`Tensor::stack` along a NEW
/// axis 1, then cast; matches `flash_attn_varlen_qkvpacked_func`'s own
/// upstream layout `crate::flash`'s module doc cites).
fn pack_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    Tensor::stack(&[q, k, v], 1)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let d = (a.to_dtype(DType::F64).unwrap() - b.to_dtype(DType::F64).unwrap())
        .unwrap()
        .abs()
        .unwrap();
    d.flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f64>()
        .unwrap()
}

fn assert_finite(t: &Tensor, what: &str) {
    let v = t
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{what}: contains a non-finite value"
    );
}

#[test]
fn o_lse_dq_dk_dv_match_the_torch_reference_within_the_analytic_tolerance() {
    let Some(dev) = cuda_device() else { return };

    for leg in LEGS {
        let q = load_f32(leg.name, "q")
            .to_device(dev.as_ref().into())
            .unwrap();
        let k = load_f32(leg.name, "k")
            .to_device(dev.as_ref().into())
            .unwrap();
        let v = load_f32(leg.name, "v")
            .to_device(dev.as_ref().into())
            .unwrap();
        let grad_out = load_f32(leg.name, "grad_out")
            .to_device(dev.as_ref().into())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let ref_o = load_f32(leg.name, "o");
        let ref_lse = load_f32(leg.name, "lse");
        let ref_dq = load_f32(leg.name, "dq");
        let ref_dk = load_f32(leg.name, "dk");
        let ref_dv = load_f32(leg.name, "dv");

        let qkv = pack_qkv(&q, &k, &v);
        let qkv_slice = qkv.flatten_all().unwrap().to_dtype(DType::BF16).unwrap();
        let cu = CuSeqlens::from_lengths(leg.lengths, &dev).unwrap();
        let cfg = VarlenConfig {
            softmax_scale: SOFTMAX_SCALE,
            window: leg.window,
            deterministic: true,
        };

        // Drive through `crate::flash` directly (see module doc) using
        // the raw `CudaSlice` the packed `qkv` tensor owns.
        let qkv_cuda = qkv_slice.storage_and_layout().0.clone(); // placeholder shape only — real access below
        drop(qkv_cuda);

        // Practical path: build the packed qkv as a raw CudaSlice via
        // candle's own device-transfer, then call the safe `flash`
        // functions with it directly (mirrors `flash_smoke.rs`'s own
        // idiom of building fixtures as host `Vec<bf16>` then
        // `dev.htod_copy`).
        let (total_q, _, num_h, _d) = qkv.dims4().unwrap();
        assert_eq!(num_h, NUM_HEADS);
        let host_qkv: Vec<bf16> = qkv.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
        let dev_raw = candle_core::cuda_backend::cudarc::driver::CudaDevice::new(0)
            .map(|_| ())
            .ok();
        let _ = dev_raw;
        let qkv_dev = dev.htod_copy(host_qkv).unwrap();

        let (o, lse) =
            jammi_kernels::flash::flash_varlen_fwd(&dev, &qkv_dev, &cu, NUM_HEADS, &cfg).unwrap();

        let o_host: Vec<bf16> = dev.dtoh_sync_copy(&o).unwrap();
        let o_tensor = Tensor::from_vec(o_host, (total_q, NUM_HEADS, HEAD_DIM), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();
        let lse_host: Vec<f32> = dev.dtoh_sync_copy(&lse).unwrap();
        let lse_tensor = Tensor::from_vec(lse_host, (NUM_HEADS, total_q), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap();

        assert_finite(&o_tensor, &format!("{}: o", leg.name));
        assert_finite(&lse_tensor, &format!("{}: lse", leg.name));

        let ref_o_max = ref_o
            .flatten_all()
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        let tol_o = analytic_tolerance(*leg.lengths.iter().max().unwrap(), ref_o_max);
        let diff_o = max_abs_diff(&o_tensor, &ref_o.to_dtype(DType::F64).unwrap());
        assert!(
            diff_o <= tol_o,
            "{}: o max abs diff {diff_o} exceeds analytic tolerance {tol_o}",
            leg.name
        );

        let ref_lse_max = ref_lse
            .flatten_all()
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        let tol_lse = analytic_tolerance(*leg.lengths.iter().max().unwrap(), ref_lse_max);
        let diff_lse = max_abs_diff(&lse_tensor, &ref_lse.to_dtype(DType::F64).unwrap());
        assert!(
            diff_lse <= tol_lse,
            "{}: lse max abs diff {diff_lse} exceeds analytic tolerance {tol_lse}",
            leg.name
        );

        // Backward.
        let do_host: Vec<bf16> = grad_out.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
        let do_dev = dev.htod_copy(do_host).unwrap();
        let d_qkv = jammi_kernels::flash::flash_varlen_bwd(
            &dev, &qkv_dev, &cu, NUM_HEADS, &o, &lse, &do_dev, &cfg,
        )
        .unwrap();
        let d_qkv_host: Vec<bf16> = dev.dtoh_sync_copy(&d_qkv).unwrap();
        let d_qkv_tensor =
            Tensor::from_vec(d_qkv_host, (total_q, 3, NUM_HEADS, HEAD_DIM), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
        let dq_t = d_qkv_tensor.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
        let dk_t = d_qkv_tensor.narrow(1, 1, 1).unwrap().squeeze(1).unwrap();
        let dv_t = d_qkv_tensor.narrow(1, 2, 1).unwrap().squeeze(1).unwrap();
        assert_finite(&dq_t, &format!("{}: dq", leg.name));
        assert_finite(&dk_t, &format!("{}: dk", leg.name));
        assert_finite(&dv_t, &format!("{}: dv", leg.name));

        for (name, got, reference) in [
            ("dq", &dq_t, &ref_dq),
            ("dk", &dk_t, &ref_dk),
            ("dv", &dv_t, &ref_dv),
        ] {
            let ref_max = reference
                .flatten_all()
                .unwrap()
                .abs()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            let tol = analytic_tolerance(*leg.lengths.iter().max().unwrap(), ref_max);
            let diff = max_abs_diff(got, &reference.to_dtype(DType::F64).unwrap());
            assert!(
                diff <= tol,
                "{}: {name} max abs diff {diff} exceeds analytic tolerance {tol}",
                leg.name
            );
        }
    }
}

/// Injection control: `softmax_scale * 2` on the jammi side (reference
/// fixture is at the CORRECT scale) must RED the O0(a) comparison — proves
/// `analytic_tolerance` actually discriminates rather than being vacuously
/// wide. One leg (`b1_s512`) is enough to prove discrimination.
#[test]
fn softmax_scale_doubling_injection_reds_the_parity_oracle() {
    let Some(dev) = cuda_device() else { return };

    let leg_name = "b1_s512";
    let lengths: &[usize] = &[512];
    let q = load_f32(leg_name, "q")
        .to_device(dev.as_ref().into())
        .unwrap();
    let k = load_f32(leg_name, "k")
        .to_device(dev.as_ref().into())
        .unwrap();
    let v = load_f32(leg_name, "v")
        .to_device(dev.as_ref().into())
        .unwrap();
    let ref_o = load_f32(leg_name, "o");

    let qkv = pack_qkv(&q, &k, &v);
    let (total_q, ..) = qkv.dims4().unwrap();
    let host_qkv: Vec<bf16> = qkv.flatten_all().unwrap().to_vec1::<bf16>().unwrap();
    let qkv_dev = dev.htod_copy(host_qkv).unwrap();
    let cu = CuSeqlens::from_lengths(lengths, &dev).unwrap();
    let bad_cfg = VarlenConfig {
        softmax_scale: SOFTMAX_SCALE * 2.0, // INJECTED — the fixture was generated at SOFTMAX_SCALE.
        window: None,
        deterministic: true,
    };

    let (o, _lse) =
        jammi_kernels::flash::flash_varlen_fwd(&dev, &qkv_dev, &cu, NUM_HEADS, &bad_cfg).unwrap();
    let o_host: Vec<bf16> = dev.dtoh_sync_copy(&o).unwrap();
    let o_tensor = Tensor::from_vec(o_host, (total_q, NUM_HEADS, HEAD_DIM), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap();
    let ref_o_max = ref_o
        .flatten_all()
        .unwrap()
        .abs()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
    let tol_o = analytic_tolerance(512, ref_o_max);
    let diff_o = max_abs_diff(&o_tensor, &ref_o.to_dtype(DType::F64).unwrap());
    assert!(
        diff_o > tol_o,
        "softmax_scale x2 injection did not RED the oracle: diff {diff_o} <= tolerance {tol_o} \
         — the comparison is not discriminating"
    );
}
