//! GPU end-to-end property suite for issue #351's GGUF/k-quant + QLoRA
//! surface (wave 18). CPU-vs-CUDA parity, a same-checkpoint quantization-loss
//! floor, an on-GPU QLoRA smoke, admission truthfulness against real device
//! memory, and a printed throughput baseline — all over a PROGRAMMATICALLY
//! written GGUF fixture (mirrors `crates/jammi-ai/tests/it/gguf_qlora.rs`'s
//! own writers: no checked-in binary `.gguf`/`.safetensors` file). That file
//! duplicated `LoraLinear`/`QuantizedLinear`-level parity ON CPU ONLY; this
//! one is the first place `model.gguf`'s resolve→load→embed path, and the
//! `FrozenBase::Quantized` QLoRA trainer, ever run their CUDA arm at all.
//!
//! Not a NEW `ci/scripts/check_gpu_parity_matrix.py` cell: every property
//! below that touches the (architecture × verb) matrix is `Bert ×
//! TextEmbedding`, already COVERED by `embeddings_parity.rs`'s
//! `gpu-parity-cell` marker — the matrix tracks (architecture × verb), not
//! weight-storage format, so a GGUF-quantized re-run of the SAME cell needs
//! no second marker (`ci/scripts/check_gpu_parity_matrix.py`'s own
//! `load_covered` tolerates more than one file naming the same cell; it is
//! not a duplicate-claim error). The QLoRA-learns and admission/throughput
//! properties are training-loop / resolver-truthfulness properties, not
//! (arch × verb) forward parity, exactly the way `fine_tune_learns.rs` (P2)
//! and `bf16_gpu_gate.rs` (P4) already document themselves as outside the
//! matrix's scope.
//!
//! Gated exactly like the rest of the suite: `live-gpu-tests` + a meaningful
//! run needs `cuda` + a visible GPU; every test early-returns with a loud
//! `tracing::warn` skip (`skip_without_gpu!`, never `#[ignore]`) otherwise.
//!
//! ## `qlora_learns_on_gpu_with_gguf_base`'s learning oracle is the held-out
//! ## val-loss curve, not the raw train-loss curve (2026-08-31)
//!
//! This test shares its exact `FineTuneConfig` (`epochs = 6`, `batch_size =
//! 8`, `warmup_steps = 0`, `lora_rank = 4`, default `LrSchedule::CosineDecay`
//! and default `early_stopping_metric = ValLoss`) with this suite's Metal
//! sibling, `crates/jammi-ai/tests/metal_quantized_gpu.rs`'s
//! `qlora_learns_on_metal_with_gguf_base` — same trainer, same fixture
//! geometry, same 4-batches/epoch training source. That file's own module
//! doc records three byte-identical Mac runs (family J determinism) in which
//! the ORIGINAL `avg_train_loss last < first` assertion was marginal-to-
//! failing (`2.856011 -> 2.856355` on one run, a rise not a decrease): with
//! only 4 batches/epoch feeding an ONLINE average, and a `CosineDecay`
//! schedule that reaches exactly `0` LR by the last of 6 epochs, that
//! average's first->last delta is dominated by near-zero-LR noise, not the
//! model's actual learning trend. `avg_val_loss` — computed once per epoch
//! over held-out data the optimizer never stepped on, always measured
//! because `early_stopping_metric` defaults to `ValLoss` — is immune to that
//! noise and decreased monotonically for 5 of the Metal sibling's 6 epochs.
//! Same trainer, same hyperparameters, same batch geometry: this CUDA arm's
//! `avg_train_loss` curve carries the identical structural noise source, so
//! this file's primary learning assertion is likewise `avg_val_loss last <
//! first` — a STRONGER oracle (the standard generalization signal a
//! held-out split exists to provide), not a loosened one. The train curve is
//! still captured and printed for the pod prove-log's baseline record, just
//! no longer trend-asserted (family K: the honest fix re-points the
//! assertion at the faithful signal; it does not touch the workload that
//! produced the noisy one).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{BackendType, LoadedModel, ModelSource, ModelTask};
use jammi_db::catalog::Catalog;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::ArtifactStore;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

// ─────────────────────────────────────────────────────────────────────────
// Fixture construction — duplicated (deliberately) from
// `tests/it/gguf_qlora.rs` rather than shared, per this file's own contract:
// small duplication into this test binary is fine; the CPU `it` suite's own
// helpers stay untouched. Geometry, seeds, and amplitude are IDENTICAL to
// that file's `small_fixture` so this file's derived tolerances below (which
// cite the fixture's own known weight amplitude) stay accurate.
// ─────────────────────────────────────────────────────────────────────────

/// FNV-1a over `name`'s bytes (family J: deterministic, no unseeded RNG) —
/// mirrors `gguf_qlora.rs::name_seed` exactly.
fn name_seed(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `n` deterministic values, amplitude `<= 0.1` — mirrors
/// `gguf_qlora.rs::det_vec` exactly (the SAME `* 0.1` scale this file's
/// [`GGUF_CUDA_ELEMENTWISE_ABS_TOL`] doc cites as the fixture's own known
/// weight amplitude).
fn det_vec(name: &str, n: usize) -> Vec<f32> {
    let seed = name_seed(name) as f64;
    (0..n)
        .map(|i| (((seed % 97.0) + 1.0) * (i as f64) * 0.037 + seed * 1e-6).sin() as f32 * 0.1)
        .collect()
}

fn det_tensor(name: &str, dims: &[usize], device: &Device) -> Tensor {
    let n: usize = dims.iter().product();
    Tensor::from_vec(det_vec(name, n), dims, device).unwrap()
}

const HIDDEN: usize = 32;
const LAYERS: usize = 1;
const HEADS: usize = 2;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 256;
const MAX_POS: usize = 128;
const TYPE_VOCAB: usize = 2;
/// The fixture's own known weight amplitude (`det_vec`'s `* 0.1` scale) —
/// used, not assumed, by [`q8_1_activation_quant_bound`]'s call sites below.
const FIXTURE_WEIGHT_AMPLITUDE: f64 = 0.1;

fn bert_tensor_map(device: &Device) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    let add = |map: &mut HashMap<String, Tensor>, name: String, dims: &[usize]| {
        let t = det_tensor(&name, dims, device);
        map.insert(name, t);
    };
    add(
        &mut map,
        "embeddings.word_embeddings.weight".into(),
        &[VOCAB, HIDDEN],
    );
    add(
        &mut map,
        "embeddings.position_embeddings.weight".into(),
        &[MAX_POS, HIDDEN],
    );
    add(
        &mut map,
        "embeddings.token_type_embeddings.weight".into(),
        &[TYPE_VOCAB, HIDDEN],
    );
    add(&mut map, "embeddings.LayerNorm.weight".into(), &[HIDDEN]);
    add(&mut map, "embeddings.LayerNorm.bias".into(), &[HIDDEN]);
    for n in 0..LAYERS {
        let p = format!("encoder.layer.{n}");
        for site in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            add(&mut map, format!("{p}.{site}.weight"), &[HIDDEN, HIDDEN]);
            add(&mut map, format!("{p}.{site}.bias"), &[HIDDEN]);
        }
        add(
            &mut map,
            format!("{p}.intermediate.dense.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        add(
            &mut map,
            format!("{p}.intermediate.dense.bias"),
            &[INTERMEDIATE],
        );
        add(
            &mut map,
            format!("{p}.output.dense.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        add(&mut map, format!("{p}.output.dense.bias"), &[HIDDEN]);
        for ln in ["attention.output.LayerNorm", "output.LayerNorm"] {
            add(&mut map, format!("{p}.{ln}.weight"), &[HIDDEN]);
            add(&mut map, format!("{p}.{ln}.bias"), &[HIDDEN]);
        }
    }
    map
}

/// The six per-layer matmul-site tensor-name prefixes — mirrors
/// `gguf_qlora.rs::bert_matmul_site_prefixes`. `INTERMEDIATE` (`in_f=32`,
/// `out_f=128`) and `output.dense` (`in_f=128`, `out_f=32`) are the two
/// widest sites; [`q8_1_activation_quant_bound`]'s call sites below use the
/// wider `in_f=128` contraction depth.
fn bert_matmul_site_prefixes() -> Vec<String> {
    let mut v = Vec::new();
    for n in 0..LAYERS {
        let p = format!("encoder.layer.{n}");
        v.push(format!("{p}.attention.self.query"));
        v.push(format!("{p}.attention.self.key"));
        v.push(format!("{p}.attention.self.value"));
        v.push(format!("{p}.attention.output.dense"));
        v.push(format!("{p}.intermediate.dense"));
        v.push(format!("{p}.output.dense"));
    }
    v
}

fn bert_config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "bert",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "type_vocab_size": TYPE_VOCAB,
        "layer_norm_eps": 1e-12,
    })
}

fn write_json(dir: &Path, name: &str, value: &serde_json::Value) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(name), serde_json::to_string(value).unwrap()).unwrap();
}

fn write_tokenizer(dir: &Path) {
    std::fs::copy(
        harness::cookbook_fixture("tiny_bert").join("tokenizer.json"),
        dir.join("tokenizer.json"),
    )
    .unwrap();
}

fn write_f32_checkpoint(dir: &Path, tensors: &HashMap<String, Tensor>) {
    std::fs::create_dir_all(dir).unwrap();
    candle_core::safetensors::save(tensors, dir.join("model.safetensors")).unwrap();
}

/// Mirrors `gguf_qlora.rs::write_gguf_checkpoint`: every matmul-site
/// `.weight` tensor is quantized at `quant`; every other tensor is written
/// as an `F32`-"quantized" `QTensor` (GGUF's lossless dense convention).
fn write_gguf_checkpoint(
    dir: &Path,
    tensors: &HashMap<String, Tensor>,
    matmul_sites: &[String],
    quant: GgmlDType,
) {
    std::fs::create_dir_all(dir).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort(); // deterministic write order (family J)
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    for name in names {
        let t = &tensors[name];
        let is_matmul_weight = matmul_sites.iter().any(|p| *name == format!("{p}.weight"));
        let dtype = if is_matmul_weight {
            quant
        } else {
            GgmlDType::F32
        };
        qtensors.push((name.clone(), QTensor::quantize(t, dtype).unwrap()));
    }
    let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    gguf_file::write(&mut writer, &[], &refs).unwrap();
}

/// Writes a fresh GGUF-quantized fixture directory (`config.json` +
/// `tokenizer.json` + `model.gguf`, quantized at [`GgmlDType::Q8_0`] — the
/// lowest-error k-quant format, matching `gguf_qlora.rs`'s own CPU-measured
/// baseline so this file's [`GGUF_VS_F32_GPU_COSINE_FLOOR`] pin is
/// comparing apples to apples) into `dir`.
fn write_q8_0_gguf_fixture(dir: &Path) {
    let device = Device::Cpu;
    let tensors = bert_tensor_map(&device);
    let sites = bert_matmul_site_prefixes();
    write_json(dir, "config.json", &bert_config_json());
    write_tokenizer(dir);
    write_gguf_checkpoint(dir, &tensors, &sites, GgmlDType::Q8_0);
}

/// Writes a fresh F32-safetensors fixture directory carrying the SAME
/// underlying tensor values `write_q8_0_gguf_fixture` quantizes — the dense
/// reference for [`gguf_on_gpu_vs_f32_safetensors_on_gpu_quantization_loss_floor`].
fn write_f32_reference_fixture(dir: &Path) {
    let device = Device::Cpu;
    let tensors = bert_tensor_map(&device);
    write_json(dir, "config.json", &bert_config_json());
    write_tokenizer(dir);
    write_f32_checkpoint(dir, &tensors);
}

/// `local:` model id for an arbitrary fixture directory (not one of
/// `harness`'s named `tests/fixtures/` / `cookbook/fixtures/` locations).
fn local_id(dir: &Path) -> String {
    format!("local:{}", dir.display())
}

/// The five-sentence text set `gguf_qlora.rs`'s own A1 test embeds, reused
/// here so this file's measured cosines are directly comparable to that
/// file's CPU-measured baseline.
const TEXTS: [&str; 5] = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "hello world",
    "gguf quantized inference test",
    "a b c d e f g",
];

// ─────────────────────────────────────────────────────────────────────────
// Oracle (1): GGUF embed parity CPU↔GPU, over the SAME quantized checkpoint.
// ─────────────────────────────────────────────────────────────────────────
//
// ## The mechanism this floor must cover (cited, not assumed)
//
// `candle-core` 0.11.0's CPU quantized forward (`QTensor::cpu_fwd`,
// `quantized/mod.rs`) dot-products the quantized `W` against `x` AT `x`'s
// OWN, UNQUANTIZED, `f32` precision — no activation-side quantization step
// at all. The CUDA forward (`QCudaStorage::fwd`, `quantized/cuda.rs:846-877`)
// instead dispatches through `fast_mmvq`/`fast_mmq`
// (`quantized/cuda.rs:364-434,435-505`), BOTH of which re-quantize the
// ACTIVATION to `Q8_1` FIRST (`quantized/cuda.rs:48-95`'s `quantize_q8_1`
// CUDA kernel launch) before the dot product — a rounding step the CPU path
// never performs. This is the EXACT mechanism
// `crates/jammi-kernels/tests/cuda_parity.rs`'s `q8_1_activation_quant_bound`
// (issue #351 wave 17) derives a per-element bound for at the SINGLE-LINEAR
// level; [`q8_1_activation_quant_bound`] below is that same formula,
// duplicated (small-duplication doctrine — the two test binaries share no
// `[dev-dependencies]` edge to hang a shared helper off).
//
// ## Why `harness::COSINE_FLOOR` (0.9999) does not apply here
//
// That floor's own doc scopes it to "a correct fp32 forward" whose ONLY
// cross-device divergence is matmul reduction-order noise. A GGUF-quantized
// forward on CUDA has a SECOND, LARGER divergence source (the Q8_1
// activation-quantization step above) that the CPU side of the SAME
// comparison never takes at all — reusing 0.9999 here would flake on a
// perfectly correct kernel, which is exactly the failure mode this file's
// own contract forbids ("never silently reuse a floor that would flake").
//
// ## The pinned floor: 0.99, not a from-scratch multi-layer derivation
//
// A rigorous ANALYTIC bound for how one linear site's Q8_1 rounding error
// propagates through the REST of a 6-matmul-site, LayerNorm+softmax+residual
// BERT layer is not tractable without the model's own Jacobian (unlike
// `quant_matmul_grad`'s single-GEMM forward, there is no single dominant
// linear map here) — naively chaining [`q8_1_activation_quant_bound`]'s
// per-site bound across all 6 sites with an L2-norm translation produces an
// error bound LARGER than the embedding vector's own norm on this tiny
// fixture (verified by hand: `2.0 * 128 * 0.1 * 0.5 / 127 ≈ 0.10` per
// element at the widest site, `× sqrt(32) ≈ 0.57` L2 per site, and even a
// SINGLE additional site already leaves no meaningful margin) — a
// mechanically "conservative" chain in that style yields a VACUOUS floor,
// not a meaningful one, which family F equally forbids. An earlier revision
// of this file pinned the SAME `[0.99, 1.0]` acceptance window
// `bf16_gpu_gate.rs` (P4) uses for its own "real device-precision mechanism,
// not a kernel bug" comparison, as a "derive now, pod-confirm later" pin.
//
// ## Tightened to `0.9999`, refuted-and-replaced by the measured pod run
// (phase-4 audit advisory, 2026-08-31)
//
// The sm_90 (H100) pod run measured `worst_cos=0.9999999987` across this
// file's five-sentence `TEXTS` set — nine nines, not the two nines the
// analytic ceiling above conservatively allowed for. Per family F9 ("a
// number is measured-and-asserted, never transcribed"), that measurement,
// not the un-pod-confirmed analytic worst case, is what pins this floor:
// `0.9999` (the SAME value `harness::COSINE_FLOOR` uses for an ordinary
// fp32 forward) leaves ~1e-4 of margin below the measured 0.9999999987 —
// roughly 1e5x the actual observed deviation from 1.0, comfortably wide
// margin while still catching a real kernel/dtype bug (which collapses
// cosine well below 0.99, per every sibling suite's own documented claim;
// see `harness.rs`'s module doc for the identical reasoning at the tighter
// number). Every assertion below still PRINTS its measured cosine
// (`--nocapture`) so a future pod run's own number is the ongoing check,
// never re-transcribed by analogy.
const GGUF_CUDA_EMBED_COSINE_FLOOR: f64 = 0.9999;

/// Headroom multiplier over [`q8_1_activation_quant_bound`]'s analytic
/// worst case — mirrors `cuda_parity.rs::Q8_1_ACTIVATION_QUANT_MARGIN`
/// exactly (same value, same justification: every term's rounding error
/// aligned in sign, never assumed to cancel).
const Q8_1_ACTIVATION_QUANT_MARGIN: f64 = 2.0;

/// Per-element worst-case absolute error a Q8_1 activation-quantization
/// step (see the module doc above) can introduce into ONE matmul site's
/// output, given a `k`-deep contraction against a weight row of magnitude
/// at most `weight_amplitude` and an activation of magnitude at most
/// `activation_amplitude`. Mirrors
/// `crates/jammi-kernels/tests/cuda_parity.rs::q8_1_activation_quant_bound`
/// byte-for-byte (duplicated per this file's own small-duplication
/// doctrine — see that function's own doc for the full derivation).
fn q8_1_activation_quant_bound(k: usize, weight_amplitude: f64, activation_amplitude: f64) -> f64 {
    let per_element_rounding_err = 0.5 * activation_amplitude / 127.0;
    Q8_1_ACTIVATION_QUANT_MARGIN * (k as f64) * weight_amplitude * per_element_rounding_err
}

/// A companion elementwise absolute-tolerance backstop (mirrors
/// `harness::ELEMENTWISE_ABS_TOL`'s role, but sized for THIS mechanism
/// rather than reused from it).
///
/// ## Re-derived in NORMALIZED units (phase-4 audit finding, 2026-08-31)
///
/// The value this tolerance is compared against
/// (`harness::max_abs_diff(&cpu_v, &gpu_v)`) is measured on
/// `encode_text_query`'s OUTPUT — an L2-NORMALIZED (unit-norm) embedding
/// vector, whose components sit at order `1/sqrt(HIDDEN)` for an
/// approximately isotropic unit vector across `HIDDEN` dims (`≈0.1768` at
/// `HIDDEN = 32`). An earlier revision of this fn plugged a RAW,
/// pre-normalization `activation_amplitude = 1.0` into
/// [`q8_1_activation_quant_bound`] — a unit-system mismatch: the resulting
/// `0.1512` was `≈85%` of a typical normalized lane's own magnitude, i.e.
/// nearly the WHOLE dynamic range a normalized component can occupy, making
/// the backstop unable to catch anything short of a near-total blowup (not
/// a meaningful per-lane guard at all — family F: a floor sized in the
/// wrong unit system cannot "bite"). Using
/// `activation_amplitude = 1/sqrt(HIDDEN)` here instead — the SAME
/// normalized-lane scale the comparison target is actually denominated in
/// — re-derives the mechanism's bound in the units it is compared against,
/// at the fixture's widest-contraction site (`output.dense`, `in_f =
/// INTERMEDIATE = 128`) and the fixture's OWN known weight amplitude
/// ([`FIXTURE_WEIGHT_AMPLITUDE`] `= 0.1`, exact): `≈0.0178`, `×1.5` headroom
/// `≈0.0267` — a real fraction (`≈15%`) of a typical normalized lane, not
/// `≈85%` of one. The sm_90 pod run measured `worst_abs=1.7568e-5` on this
/// exact comparison — `≈1500x` below `0.0267` — so this re-derived bound is
/// both unit-correct AND comfortably wide-margin over the observed value;
/// `10x` that measured anchor (`≈1.7568e-4`) is kept as an explicit floor so
/// the analytic term is never trusted to dominate by construction alone.
/// `10x` the sm_90-measured `worst_abs` anchor (`1.7568e-5`, 2026-08-31) —
/// a measured-with-margin floor (family F9), not an arbitrary round number.
const GGUF_CUDA_ELEMENTWISE_MEASURED_ANCHOR_FLOOR: f64 = 1.7568e-4;

fn gguf_cuda_elementwise_abs_tol() -> f64 {
    let normalized_lane_amplitude = 1.0 / (HIDDEN as f64).sqrt();
    let single_site_bound = q8_1_activation_quant_bound(
        INTERMEDIATE,
        FIXTURE_WEIGHT_AMPLITUDE,
        normalized_lane_amplitude,
    );
    (single_site_bound * 1.5).max(GGUF_CUDA_ELEMENTWISE_MEASURED_ANCHOR_FLOOR)
}

#[tokio::test(flavor = "multi_thread")]
async fn gguf_embedding_cpu_gpu_parity_within_q8_1_activation_quant_floor() {
    skip_without_gpu!();
    harness::loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    let model = local_id(&gguf_dir);

    let cpu_dir = TempDir::new().unwrap();
    let cpu = harness::cpu_session(cpu_dir.path()).await;
    let gpu_dir = TempDir::new().unwrap();
    let gpu = harness::gpu_session(gpu_dir.path()).await;

    let elementwise_tol = gguf_cuda_elementwise_abs_tol();
    let mut total_values = 0usize;
    let mut finite_values = 0usize;
    let mut worst_cos = 1.0f64;
    let mut worst_abs = 0.0f64;
    for text in TEXTS {
        let cpu_v = cpu.encode_text_query(&model, text).await.unwrap();
        let gpu_v = gpu.encode_text_query(&model, text).await.unwrap();
        assert_eq!(
            cpu_v.len(),
            gpu_v.len(),
            "CPU and GPU GGUF query vectors must share a dimension"
        );
        for &v in cpu_v.iter().chain(gpu_v.iter()) {
            total_values += 1;
            if v.is_finite() {
                finite_values += 1;
            }
        }
        let cos = harness::cosine(&cpu_v, &gpu_v);
        let abs = harness::max_abs_diff(&cpu_v, &gpu_v);
        tracing::info!(text, cos, abs, "GGUF CPU↔GPU embed parity");
        worst_cos = worst_cos.min(cos);
        worst_abs = worst_abs.max(abs);
    }

    // F9: every value finite BY COUNT, never a vacuous "some finite" pass.
    assert_eq!(
        finite_values, total_values,
        "expected every GGUF CPU/GPU embedding value finite, got {finite_values}/{total_values}"
    );

    eprintln!(
        "gguf_embedding_cpu_gpu_parity: worst_cos={worst_cos} floor={GGUF_CUDA_EMBED_COSINE_FLOOR} \
         worst_abs={worst_abs} elementwise_tol={elementwise_tol} \
         (Q8_1 activation-quantization mechanism, margin={Q8_1_ACTIVATION_QUANT_MARGIN})"
    );
    assert!(
        worst_cos >= GGUF_CUDA_EMBED_COSINE_FLOOR,
        "GGUF CPU↔GPU worst-case cosine {worst_cos} below the Q8_1-activation-quantization \
         floor {GGUF_CUDA_EMBED_COSINE_FLOOR} — a real kernel/dtype bug, not quantization noise"
    );
    assert!(
        worst_abs <= elementwise_tol,
        "GGUF CPU↔GPU worst-case |Δ| {worst_abs} exceeds the derived tolerance \
         {elementwise_tol} — a real kernel/dtype bug, not quantization noise"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (2): GGUF-on-GPU vs f32-safetensors-on-GPU, same underlying
// weights — the quantization-LOSS floor (Q8_0 weight rounding only; BOTH
// arms run on the same device, so the Q8_1 activation-quantization
// mechanism above does not enter this comparison at all).
// ─────────────────────────────────────────────────────────────────────────

/// `tests/it/gguf_qlora.rs::gguf_embedding_matches_f32_reference_within_a_measured_cosine_floor`
/// measured, on this workspace's hermetic CPU dev/CI arm (2026-08-30,
/// re-confirmed 2026-08-31), `mean_cosine=0.99999964, min_cosine=0.9999995`
/// for a Q8_0-quantized 1-layer/32-dim BERT tower vs its F32 reference over
/// this SAME five-sentence set. This is the CPU-side proof that Q8_0's OWN
/// weight-quantization loss (independent of any device) is tiny. This test
/// re-runs the identical comparison with BOTH arms on the GPU instead of CPU.
///
/// ## Tightened to `0.9999`, refuted-and-replaced by the measured pod run
/// (phase-4 audit advisory, 2026-08-31)
///
/// An earlier revision of this file pinned a `0.999` floor here — un-pod-
/// confirmed at the time. The sm_90 (H100) pod run measured
/// `worst_cos=0.9999995` on this exact GPU-vs-GPU comparison, matching the
/// CPU-hermetic measurement above almost exactly (as expected: Q8_0 weight-
/// quantization loss is a device-independent property of the quantized
/// bytes themselves). Per family F9, that measurement — not the earlier
/// un-confirmed pin — now sets the floor: `0.9999` leaves real margin
/// (`≈1e-4`) below the measured `0.9999995` while staying decisively above
/// what a real GPU-side quantization bug (a wrong dequantize path, wrong
/// dtype) would produce — this assertion's job is to catch that bug, not to
/// re-derive the loss bound. The measured value is still PRINTED every run
/// so the pod prove-log keeps recording the on-device baseline for future
/// tightening.
const GGUF_VS_F32_GPU_COSINE_FLOOR: f64 = 0.9999;

#[tokio::test(flavor = "multi_thread")]
async fn gguf_on_gpu_vs_f32_safetensors_on_gpu_quantization_loss_floor() {
    skip_without_gpu!();
    harness::loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    let f32_dir = tmp.path().join("f32_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    write_f32_reference_fixture(&f32_dir);

    let gguf_session_dir = TempDir::new().unwrap();
    let gguf_gpu = harness::gpu_session(gguf_session_dir.path()).await;
    let f32_session_dir = TempDir::new().unwrap();
    let f32_gpu = harness::gpu_session(f32_session_dir.path()).await;

    let gguf_model = local_id(&gguf_dir);
    let f32_model = local_id(&f32_dir);

    let mut worst_cos = 1.0f64;
    let mut mean_cos = 0.0f64;
    for text in TEXTS {
        let gguf_v = gguf_gpu.encode_text_query(&gguf_model, text).await.unwrap();
        let f32_v = f32_gpu.encode_text_query(&f32_model, text).await.unwrap();
        assert_eq!(gguf_v.len(), f32_v.len(), "must share a dimension");
        let cos = harness::cosine(&gguf_v, &f32_v);
        tracing::info!(
            text,
            cos,
            "GGUF-on-GPU vs f32-on-GPU quantization-loss cosine"
        );
        worst_cos = worst_cos.min(cos);
        mean_cos += cos / TEXTS.len() as f64;
    }

    eprintln!(
        "gguf_on_gpu_vs_f32_safetensors_on_gpu: worst_cos={worst_cos} mean_cos={mean_cos} \
         floor={GGUF_VS_F32_GPU_COSINE_FLOOR} (device baseline for the pod prove log)"
    );
    assert!(
        worst_cos > GGUF_VS_F32_GPU_COSINE_FLOOR,
        "GGUF-on-GPU vs f32-on-GPU worst-case cosine {worst_cos} at or below the \
         CPU-measured-and-pinned floor {GGUF_VS_F32_GPU_COSINE_FLOOR} — either the GPU \
         dequantize/dtype path is broken, or the CPU-measured floor no longer holds on device"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (3): QLoRA-on-GPU smoke, GGUF base — mirrors `fine_tune_learns.rs`
// (P2) exactly, with a GGUF base model instead of `tiny_bert`'s
// safetensors. QLoRA activates purely off the base artifact
// (`fine_tune/worker.rs`'s `is_gguf` switch — no new trainer/config knob),
// so this exercises `FrozenBase::Quantized` end-to-end on CUDA for the
// first time. Not a `check_gpu_parity_matrix.py` cell (same reasoning as
// `fine_tune_learns.rs`'s own doc: a training-loop property, no CPU
// baseline).
// ─────────────────────────────────────────────────────────────────────────

async fn add_training_source(session: &Arc<jammi_ai::session::InferenceSession>) {
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(harness::fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn qlora_learns_on_gpu_with_gguf_base() {
    skip_without_gpu!();
    harness::loss_capture::install();
    harness::loss_capture::reset();

    let fixture_dir = TempDir::new().unwrap();
    let gguf_dir = fixture_dir.path().join("gguf_base");
    write_q8_0_gguf_fixture(&gguf_dir);
    let model = local_id(&gguf_dir);

    let dir = TempDir::new().unwrap();
    let session = harness::gpu_session(dir.path()).await;
    add_training_source(&session).await;
    let _worker = jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(&session)
        .expect("default worker intervals are valid");

    let job = session
        .fine_tune(
            "training",
            &model,
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(FineTuneConfig {
                epochs: 6, // >=2 so first->last carries a decrease signal
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // (a) completes on the GPU, over a Quantized (GGUF) frozen base.
    job.wait().await.unwrap();
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "GPU QLoRA job should complete, got {}",
        record.status
    );

    // (b) PRIMARY learning assertion: held-out avg_val_loss decreases
    // first->last epoch — the faithful signal at this suite's
    // hyperparameters. See this file's module doc for the traced mechanism
    // (a 4-batches/epoch online train-loss average, `warmup_steps = 0`, and
    // an LR schedule decaying to exactly 0 by the last epoch, measured on
    // the Metal sibling to be dominated by near-zero-LR noise rather than a
    // real learning trend). `avg_train_loss` is still captured and printed
    // below purely as a baseline record for the pod prove-log, never
    // trend-asserted.
    let train_curve = harness::loss_capture::captured();
    let val_curve = harness::loss_capture::captured_val();
    eprintln!(
        "qlora_learns_on_gpu_with_gguf_base: train_curve={train_curve:?} (printed as a \
         baseline record only, NOT trend-asserted) val_curve={val_curve:?}"
    );

    // Every reported loss (train AND val) finite, by count (family F9).
    harness::assert_all_finite("qlora_gguf", &[&train_curve, &val_curve]);

    let (first, last) = harness::assert_loss_decreases("qlora_gguf_val_loss", &val_curve);

    // (c) the adapter changes embeddings vs the (quantized) base model —
    // the LoRA delta is non-zero, i.e. training genuinely moved the
    // adapters, not merely completed a no-op run.
    let models = session.catalog().list_models().await.unwrap();
    let ft = models
        .iter()
        .find(|m| m.model_id.starts_with("jammi:fine-tuned:"))
        .expect("fine-tuned model registered");
    let ft_name = ft.model_id.split("::").next().unwrap();

    let base = session
        .encode_text_query(&model, "quantum computing")
        .await
        .unwrap();
    let tuned = session
        .encode_text_query(ft_name, "quantum computing")
        .await
        .unwrap();
    let delta: f32 = base.iter().zip(&tuned).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        delta > 1e-6,
        "GPU QLoRA-trained adapter must change embeddings (LoRA delta non-zero), delta={delta}"
    );

    tracing::info!(
        first_val_loss = first,
        last_val_loss = last,
        embed_delta = delta,
        "QLoRA learns on GPU over a GGUF base"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (4): admission truthfulness on hardware — the resolver's
// `estimated_memory` compared against ACTUAL device memory delta, measured
// via `nvidia-smi` (candle 0.11 exposes no allocator-stats API).
// ─────────────────────────────────────────────────────────────────────────

/// Poll total device memory in use, in bytes, via `nvidia-smi` — mirrors
/// `crates/jammi-bench/src/finetune_step.rs::device_memory_used_bytes`
/// exactly (duplicated: `jammi-bench` depends on `jammi-ai`, not the
/// reverse, so that implementation is unreachable here — see this file's
/// own small-duplication doctrine). Whole-device, not per-process: the pod
/// prove-lane session is this device's only consumer.
fn device_memory_used_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|mib| mib * 1024 * 1024)
}

/// [`device_memory_used_bytes`], or a hard failure when `JAMMI_REQUIRE_CUDA`
/// is set and the reading is unavailable. `skip_without_gpu!` already
/// requires a real CUDA device on this path; on the pod session that is
/// SUPPOSED to have one (`JAMMI_REQUIRE_CUDA=1`), `nvidia-smi` being
/// unusable is also a hard failure, never a silent skip — the same
/// require-gate idiom `crates/jammi-ai/src/fine_tune/optimizer.rs::
/// cuda_device` and `crates/jammi-bench/src/finetune_step.rs::
/// vram_probe_present`'s callers carry for device-measurement channels.
fn device_memory_used_bytes_or_require(test: &str) -> Option<u64> {
    match device_memory_used_bytes() {
        Some(v) => Some(v),
        None => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_CUDA is set but nvidia-smi is unavailable — cannot \
                     measure device memory; a silent skip is not acceptable here"
                );
            }
            None
        }
    }
}

/// One CUDA caching-allocator pool block, per
/// `crates/jammi-bench/src/finetune_step.rs`'s own documented convention
/// (also recorded in `crates/jammi-kernels/artifacts/cuda-runs/
/// 2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json`'s `_comment`: the
/// allocator rounds allocations up to ~32 MiB blocks, so a `nvidia-smi`-
/// measured before/after delta is quantized to this granularity, never a
/// byte-exact live-allocation figure). The admission allowance below is
/// EXACTLY one block — ALLOCATOR GRANULARITY ONLY, derived from and cited
/// to this same convention, never a second, blind block of slop stacked on
/// top (the phase-4 audit's own finding on this oracle: a 64 MiB, two-block
/// allowance against a 166 KB `estimated_memory` made the pass arm vacuous
/// — literally any non-negative estimate would clear it regardless of
/// truthfulness).
const ALLOCATOR_POOL_BLOCK_BYTES: u64 = 32 * 1024 * 1024;
const ADMISSION_ALLOWANCE_BYTES: u64 = ALLOCATOR_POOL_BLOCK_BYTES;

/// PHASE 0 of the two-phase measurement below: touch the CUDA device with a
/// small, non-quantized allocation and synchronize, before this oracle's
/// `before` snapshot is taken. This is the fix for the audit's other named
/// mechanism ("the fail arm charges CUDA context/pool overhead to the
/// resolver"): context/stream/allocator-pool bring-up is a one-time,
/// per-process, per-device cost paid on FIRST device use, not a per-load
/// cost `estimated_memory` has any duty to predict — measuring `before`
/// prior to ANY device touch charges that bring-up cost into the GGUF-load
/// delta this oracle is supposed to isolate. Settling first means `after -
/// before` isolates the load-specific allocation instead.
fn settle_cuda_device(device: &Device) {
    let warm = Tensor::zeros((64, 64), candle_core::DType::F32, device).unwrap();
    let _ = warm.sum_all().unwrap();
    device.synchronize().unwrap();
}

fn gpu_device_config() -> DeviceConfig {
    DeviceConfig {
        gpu_device: 0,
        memory_fraction: 1.0,
        require_gpu: true,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    }
}

fn ephemeral_artifact_store() -> Arc<ArtifactStore> {
    let cache = tempfile::tempdir().unwrap().keep();
    Arc::new(
        ArtifactStore::with_root(
            jammi_db::storage::StorageUrl::memory("gguf-quantized-gpu-test-artifacts"),
            jammi_db::storage::StorageRegistry::new(),
            cache,
        )
        .unwrap(),
    )
}

/// Two-phase measurement (phase-4 audit finding 1's fix). What this oracle
/// CAN isolate: the device-memory delta specifically attributable to
/// resolving+loading THIS GGUF checkpoint, on an already-settled CUDA
/// context (Phase 0 below excludes context/stream/pool bring-up). What it
/// CANNOT isolate: sub-block allocator rounding — `nvidia-smi`'s whole-
/// device reading is quantized to [`ALLOCATOR_POOL_BLOCK_BYTES`], so this
/// oracle can never prove byte-exact truthfulness, only that the estimate
/// does not under-report by more than one block of allocator slop.
#[tokio::test(flavor = "multi_thread")]
async fn gguf_gpu_load_admission_estimate_is_truthful_against_measured_device_memory() {
    skip_without_gpu!();
    harness::loss_capture::install();

    // Phase 0: settle the CUDA device (context/stream/pool bring-up paid)
    // BEFORE the `before` snapshot — see `settle_cuda_device`'s own doc.
    let cuda = Device::new_cuda(0).unwrap();
    settle_cuda_device(&cuda);

    let Some(before) = device_memory_used_bytes_or_require(
        "gguf_gpu_load_admission_estimate_is_truthful_against_measured_device_memory",
    ) else {
        tracing::warn!("SKIP: nvidia-smi unavailable — cannot measure device memory delta");
        return;
    };

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    write_q8_0_gguf_fixture(&gguf_dir);

    let catalog_dir = TempDir::new().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, ephemeral_artifact_store()).unwrap();
    let source = ModelSource::local(gguf_dir.as_path());
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, Some(BackendType::Candle))
        .await
        .unwrap();

    // Phase 1: resolve+load the GGUF model on the SAME already-settled
    // device, then synchronize. `backend.load` and `cuda.synchronize()` run
    // with NO `.await` between them, so both execute on the same OS thread
    // regardless of the tokio runtime's flavor — candle's `CudaDevice::new`
    // binds `context.per_thread_stream()` (`cuda_backend/device.rs:420`), a
    // genuinely THREAD-LOCAL stream, so reusing the SAME already-constructed
    // `cuda` handle here still drains the SAME stream `backend.load`
    // enqueued its work on.
    let backend = CandleBackend;
    let loaded: LoadedModel = backend.load(&resolved, &gpu_device_config()).unwrap();
    cuda.synchronize().unwrap();

    // AFTER snapshot MUST run with the model still resident — this is the
    // audit's other named bug: the old `drop(loaded)` ran BEFORE this
    // snapshot despite its own comment claiming to "keep the model resident
    // through the synchronize/measure window". `drop(loaded)` now runs
    // AFTER `after` is captured.
    let Some(after) = device_memory_used_bytes_or_require(
        "gguf_gpu_load_admission_estimate_is_truthful_against_measured_device_memory",
    ) else {
        tracing::warn!(
            "SKIP: nvidia-smi unavailable after load — cannot measure device memory delta"
        );
        return;
    };
    drop(loaded);

    // Signed so a genuine DECREASE (memory freed elsewhere during the
    // measurement window — never a legitimate outcome of this load-only
    // window) is distinguishable from a zero delta. `saturating_sub` on the
    // unsigned reading would silently fold both into 0, hiding a real bug
    // behind the same "granularity artifact" story a true zero gets below.
    let raw_delta = after as i64 - before as i64;
    let estimated = resolved.estimated_memory as u64;

    eprintln!(
        "gguf_gpu_admission_truthfulness: estimated_memory={estimated} raw_delta={raw_delta} \
         (before={before} after={after}) allowance={ADMISSION_ALLOWANCE_BYTES} (one allocator \
         pool block)"
    );

    // Hard failure (audit advisory 5): `nvidia-smi`'s pool-block
    // quantization can only round a small positive delta DOWN toward zero —
    // it cannot manufacture a negative one. A negative `raw_delta` here
    // means device memory was measurably freed during a window that only
    // ever loads a model, which is impossible for a correct measurement and
    // must fail loud rather than be laundered into the soft zero-delta skip
    // below.
    assert!(
        raw_delta >= 0,
        "gguf_gpu_admission_truthfulness: measured device memory DECREASED across the load \
         window (before={before} after={after}, raw_delta={raw_delta}) — this is not a \
         granularity artifact (quantization can only round toward zero, never negative); \
         something else freed device memory during measurement, or the before/after ordering \
         is wrong"
    );

    // Soft skip (audit advisory 5), same shape as this test's two
    // nvidia-smi-unavailable arms above: `nvidia-smi`'s whole-device
    // reading is quantized to `ALLOCATOR_POOL_BLOCK_BYTES`, so a genuinely
    // small GGUF load can round down to an observed delta of exactly zero.
    // That is a measurement-granularity artifact, not evidence
    // `estimated_memory` is untruthful — assert nothing about it and skip.
    if raw_delta == 0 {
        // Re-verify the measurement channel is still healthy before treating
        // a zero delta as an honest granularity artifact rather than a
        // silently degraded `nvidia-smi` read: routes through the SAME
        // require-gated helper the before/after snapshots use, so
        // `JAMMI_REQUIRE_CUDA` still turns a genuinely broken channel into a
        // hard failure here too, never a laundered skip.
        let _ = device_memory_used_bytes_or_require(
            "gguf_gpu_load_admission_estimate_is_truthful_against_measured_device_memory: \
             zero-delta re-check",
        );
        tracing::warn!(
            "SKIP: measured device-memory delta was 0 (before={before} after={after}) — \
             nvidia-smi's whole-device reading is quantized to \
             {ADMISSION_ALLOWANCE_BYTES}-byte allocator pool blocks and can round a genuinely \
             small GGUF load down to zero; cannot say whether estimated_memory={estimated} is \
             truthful"
        );
        return;
    }
    let measured_delta = raw_delta as u64;

    // Direction-only, wide-documented-bound assertion (F9): the resolver's
    // estimate must not UNDER-report true residency by more than the
    // allocator's own single-pool-block granularity — never a byte-exact
    // equality `nvidia-smi`'s whole-device, pool-quantized reading cannot
    // support.
    assert!(
        estimated + ADMISSION_ALLOWANCE_BYTES >= measured_delta,
        "resolver estimated_memory {estimated} under-reports the measured device memory \
         delta {measured_delta} by more than the {ADMISSION_ALLOWANCE_BYTES}-byte allocator-\
         pool-block allowance — the admission gate is not truthful on real hardware"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (5): throughput baseline — printed only, no assertion. Perf
// assertions belong to the perf-claims machinery (`ci/scripts/perf/`), not
// this correctness suite.
// ─────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn gguf_vs_f32_gpu_throughput_baseline() {
    skip_without_gpu!();
    harness::loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    let f32_dir = tmp.path().join("f32_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    write_f32_reference_fixture(&f32_dir);

    let gguf_session_dir = TempDir::new().unwrap();
    let gguf_gpu = harness::gpu_session(gguf_session_dir.path()).await;
    let f32_session_dir = TempDir::new().unwrap();
    let f32_gpu = harness::gpu_session(f32_session_dir.path()).await;

    let gguf_model = local_id(&gguf_dir);
    let f32_model = local_id(&f32_dir);

    const ROWS: usize = 20;
    let rows: Vec<&str> = TEXTS.iter().cycle().take(ROWS).copied().collect();

    let t0 = Instant::now();
    for &text in &rows {
        let _ = gguf_gpu.encode_text_query(&gguf_model, text).await.unwrap();
    }
    let gguf_elapsed = t0.elapsed();

    let t0 = Instant::now();
    for &text in &rows {
        let _ = f32_gpu.encode_text_query(&f32_model, text).await.unwrap();
    }
    let f32_elapsed = t0.elapsed();

    let gguf_rows_per_sec = ROWS as f64 / gguf_elapsed.as_secs_f64();
    let f32_rows_per_sec = ROWS as f64 / f32_elapsed.as_secs_f64();
    eprintln!(
        "gguf_vs_f32_gpu_throughput_baseline: rows={ROWS} \
         gguf_rows_per_sec={gguf_rows_per_sec:.2} (elapsed={gguf_elapsed:?}) \
         f32_rows_per_sec={f32_rows_per_sec:.2} (elapsed={f32_elapsed:?}) \
         -- printed baseline only, no assertion (perf-claims machinery owns thresholds)"
    );
}
