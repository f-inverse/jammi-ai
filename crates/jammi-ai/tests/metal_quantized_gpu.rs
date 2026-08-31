//! Metal end-to-end property suite for issue #351's GGUF/k-quant + QLoRA
//! surface (wave 18, Half B). This is the Metal-arm sibling of
//! `tests/gpu_capability/gguf_quantized_gpu.rs` (Half A, CUDA-gated) — a
//! SEPARATE test binary rather than a `gpu_capability` module because that
//! suite's `harness.rs` hardcodes CUDA (`Device::new_cuda`, a CUDA-only
//! driver/compute-capability admission gate); Metal needs neither, and
//! reusing that harness would mean threading a device-kind enum through code
//! that has no third case today. Required feature is `metal` alone (not
//! `live-gpu-tests`): `metal = ["local", ...]` already implies the engine, and
//! this binary's own `skip_without_gpu!` guard is the meaningful-run gate, not
//! a separate opt-in knob.
//!
//! Fixture construction (the GGUF/f32 checkpoint writers) is DELIBERATELY
//! duplicated from `gguf_quantized_gpu.rs` (itself duplicated from
//! `tests/it/gguf_qlora.rs`) rather than shared: these are three independent
//! test binaries with no `[dev-dependencies]` edge between them to hang a
//! shared helper off, and this file's own contract (like its two siblings')
//! is that small duplication into a test binary is fine.
//!
//! ## Why this file's CPU↔Metal embed-parity floor is NOT Half A's 0.99
//!
//! Half A's `GGUF_CUDA_EMBED_COSINE_FLOOR = 0.99` exists because CUDA's
//! quantized matmul (`QCudaStorage::fwd`, `quantized/cuda.rs:846-877`)
//! re-quantizes the ACTIVATION to `Q8_1` before the dot product
//! (`quantize_q8_1`, `quantized/cuda.rs:48-95`) — a second, GPU-only rounding
//! step the CPU path never takes. Metal's quantized matmul has NO such step:
//! `QMetalStorage::fwd` (`candle-core-0.11.0/src/quantized/metal.rs:344-390`)
//! asserts `storage.dtype() == DType::F32` on its activation input at line
//! 390 (the `n > 1` / non-`fwd_mv` arm; the `n == 1` `fwd_mv` arm at line
//! 282-342 carries no such assert but still only ever receives an `F32`
//! buffer in this engine — see the next paragraph), meaning the GPU kernel
//! dot-products the quantized weight against the activation AT ITS OWN,
//! UNQUANTIZED, `f32` precision — structurally the SAME computation
//! `QTensor::cpu_fwd` performs on CPU, just a different reduction order /
//! kernel implementation (a fused per-block dequant-then-dot GPU kernel vs a
//! CPU element loop). `jammi_lora::frozen_base::QuantizedLinear::forward`'s
//! own "uniform F32 activation rule" doc (`crates/jammi-lora/src/
//! frozen_base.rs:64-77`) names that exact Metal assert as ITS reason for
//! existing: casting the activation to `F32` unconditionally, on every
//! device, before calling into `quant_matmul_grad`, is "the ONE choice that
//! can never reach that panic" — so this engine's Metal quantized forward
//! never sees a non-`F32` activation, on any call path, by construction. That
//! makes this comparison the SAME divergence category as an ordinary fp32
//! CPU↔GPU parity check (reduction-order noise only), not Half A's
//! Q8_1-activation-quantization category — so this file borrows the ordinary
//! fp32 floor's REASONING, not Half A's number.
//!
//! ## Why the floor is still measured, not `0.9999`-by-analogy
//!
//! The dequant-then-dot Metal kernel is nonetheless a genuinely different
//! ALGORITHM from `gpu_capability`'s plain dense fp32 matmul comparison (that
//! floor's own justification is scoped to "a correct fp32 forward" with no
//! quantization involved at all) — a per-block dequantization step
//! introduces its own small rounding before the dot product even starts.
//! Rather than assume that stays under `1e-4` of cosine slack by analogy,
//! [`GGUF_METAL_EMBED_COSINE_FLOOR`] is pinned from a value ACTUALLY MEASURED
//! on this Mac's Metal device (see the constant's own doc for the measured
//! number and date) with real headroom under it — family F: a number is
//! measured-and-asserted, never transcribed by analogy from a different
//! kernel's floor.
//!
//! ## The admission-truthfulness oracle (Half A's Oracle 4) is DELIBERATELY
//! ## ABSENT here — an honest omission, not an oversight
//!
//! Half A measures the resolver's `estimated_memory` against a real
//! `nvidia-smi`-reported device-memory delta. macOS/Metal has no
//! non-privileged equivalent: candle 0.11's `MetalDevice` exposes no
//! allocator-stats API (grepped; none), and Apple Silicon's unified-memory
//! architecture means there is no separate "device memory" counter a CLI
//! tool reports the way `nvidia-smi` reports discrete VRAM — the closest
//! analogues (`powermetrics`, `ioreg` GPU counters) either need `sudo` or
//! report OS-level RSS that a live `tokio` test process cannot use as a
//! clean device-memory oracle. Faking this oracle with a vacuous
//! whole-process RSS check would violate family F's non-vacuous-control
//! requirement more than simply not shipping it — the honest choice (family
//! K) is to state the gap here rather than assert something unfalsifiable.
//!
//! ## KNOWN, LIVE FAILURE: `qlora_learns_on_metal_with_gguf_base` (2026-08-31)
//!
//! This oracle currently FAILS on real Metal hardware, and this is a
//! genuine product gap, not a test-writing bug — left asserting real success
//! (not weakened, not `#[ignore]`d) so the failure stays visible rather than
//! silently dropped. Root cause, traced to the exact mechanism (family F —
//! not merely observed, the causal chain is walked to its source):
//!
//! `LoraLinear::forward_composed` (`crates/jammi-lora/src/lora_linear.rs:
//! 834-838`) unconditionally routes any non-zero-dropout training forward
//! through `jammi_kernels::ops::DropoutFused` (`apply1(&x_lora,
//! DropoutFused::new(..))`) — a `CustomOp1` implemented for CPU and CUDA
//! only (`crates/jammi-kernels/src/ops/dropout.rs`; no `metal_fwd` arm
//! exists). `FineTuneConfig::lora_dropout` defaults to `0.05`
//! (`crates/jammi-wire/src/fine_tune.rs:491`), so `forward_composed` is
//! reached with `dropout_key = Some(..)` on every default-config LoRA
//! training forward. `FrozenBase::Quantized` (QLoRA) ALWAYS routes through
//! `forward_composed` — the fused-kernel training arm's own domain
//! (`lora_linear_admission_predicate`) is Dense-weight-only and is never
//! even offered a quantized base (`lora_linear.rs:735-751`) — so EVERY QLoRA
//! training forward on Metal hits this gap unconditionally. This is broader
//! than QLoRA, too: a *dense* LoRA base whose own fused-kernel domain check
//! declines (`lora_linear_admission_predicate`'s `device_is_cpu_or_cuda`
//! predicate, `lora_linear.rs:253`, is false on Metal by construction) falls
//! back to the SAME `forward_composed` path, so ordinary (non-quantized)
//! LoRA training with the default `lora_dropout` is ALSO broken on Metal —
//! this suite only exercises the QLoRA arm, but the mechanism is shared.
//! `jammi-lora` / `jammi-kernels` are outside this suite's owning crate
//! (`jammi-ai`), so the fix (a `metal_fwd` arm for `DropoutFused`, or a
//! device-agnostic composed-fallback dropout that does not depend on the
//! fused CustomOp) is not made here — this file documents the measured,
//! traced failure as the honest result of actually running the suite on
//! real hardware (family K).
//!
//! Gated exactly like the rest of the GPU suites: every test early-returns
//! with a loud `tracing::warn` skip (`skip_without_gpu!`, never `#[ignore]`)
//! when no Metal device is usable.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, InferenceConfig, JammiConfig, LoggingConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_numerics::ComputePrecision;
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────
// Metal-availability skip guard — mirrors `gpu_capability/harness.rs`'s CUDA
// guard exactly, substituting `Device::new_metal` for `Device::new_cuda`.
// Metal has no driver/compute-capability admission floor to duplicate: candle
// 0.11's Metal backend carries no analogous JIT-version or architecture gate.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "metal")]
fn gpu_available() -> bool {
    Device::new_metal(0).is_ok()
}

#[cfg(not(feature = "metal"))]
fn gpu_available() -> bool {
    false
}

macro_rules! skip_without_gpu {
    () => {{
        if !gpu_available() {
            tracing::warn!(
                "SKIP: no usable Metal device (build with `--features metal` on a Mac with a \
                 Metal GPU to run this suite)"
            );
            return;
        }
    }};
}

// ─────────────────────────────────────────────────────────────────────────
// Session builders — mirrors `gpu_capability/harness.rs::{cpu_session,
// gpu_session}`, duplicated per this file's own small-duplication doctrine.
// ─────────────────────────────────────────────────────────────────────────

fn config_for(artifact_dir: &Path, device: i32) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device,
            require_gpu: device >= 0,
            compute_precision: ComputePrecision::F32,
            ..Default::default()
        },
        inference: InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: LoggingConfig {
            level: "info".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

async fn cpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, -1))
            .await
            .expect("cpu-pinned session"),
    )
}

/// Build a Metal-pinned (`gpu.device = 0`, `require_gpu = true`) session.
/// Only call after [`gpu_available`] / `skip_without_gpu!`.
async fn gpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, 0))
            .await
            .expect("metal-pinned session (require_gpu=true)"),
    )
}

// ─────────────────────────────────────────────────────────────────────────
// Parity comparison helpers — mirrors `gpu_capability/harness.rs` exactly.
// ─────────────────────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "parity vectors must share a dimension");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "parity vectors must share a dimension");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────
// Per-epoch loss capture — mirrors `gpu_capability/harness.rs::loss_capture`
// exactly (same mechanism: one process-global `tracing` subscriber records
// each fine-tune run's `(epoch, avg_train_loss)` "Epoch complete" events).
// ─────────────────────────────────────────────────────────────────────────

mod loss_capture {
    use std::sync::{Mutex, OnceLock};

    use tracing::field::{Field, Visit};
    use tracing::Event;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer;

    static EPOCHS: OnceLock<Mutex<Vec<(u64, f64)>>> = OnceLock::new();
    static INSTALLED: OnceLock<()> = OnceLock::new();

    fn buffer() -> &'static Mutex<Vec<(u64, f64)>> {
        EPOCHS.get_or_init(|| Mutex::new(Vec::new()))
    }

    struct EpochLossLayer;

    struct EpochVisitor {
        epoch: Option<u64>,
        loss: Option<f64>,
        is_epoch_event: bool,
    }

    impl Visit for EpochVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" && format!("{value:?}").contains("Epoch complete") {
                self.is_epoch_event = true;
            }
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            if field.name() == "epoch" {
                self.epoch = Some(value);
            }
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            if field.name() == "epoch" {
                self.epoch = Some(value as u64);
            }
        }
        fn record_f64(&mut self, field: &Field, value: f64) {
            if field.name() == "avg_train_loss" {
                self.loss = Some(value);
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for EpochLossLayer {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut v = EpochVisitor {
                epoch: None,
                loss: None,
                is_epoch_event: false,
            };
            event.record(&mut v);
            if v.is_epoch_event {
                if let (Some(e), Some(l)) = (v.epoch, v.loss) {
                    buffer().lock().unwrap().push((e, l));
                }
            }
        }
    }

    pub fn install() {
        INSTALLED.get_or_init(|| {
            use tracing_subscriber::EnvFilter;
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let fmt = tracing_subscriber::fmt::layer()
                .with_test_writer()
                .with_target(false);
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt)
                .with(EpochLossLayer)
                .init();
        });
    }

    pub fn reset() {
        buffer().lock().unwrap().clear();
    }

    pub fn captured() -> Vec<(u64, f64)> {
        buffer().lock().unwrap().clone()
    }
}

fn assert_loss_decreases(label: &str, curve: &[(u64, f64)]) -> (f64, f64) {
    assert!(
        curve.len() >= 2,
        "{label}: need >=2 epochs to prove a loss decrease, captured {curve:?}"
    );
    let first = curve.first().unwrap().1;
    let last = curve.last().unwrap().1;
    tracing::info!(label, first, last, epochs = curve.len(), "loss curve");
    assert!(
        first.is_finite() && last.is_finite(),
        "{label}: non-finite loss in {curve:?}"
    );
    assert!(
        last < first,
        "{label}: training loss did not decrease on GPU (first {first}, last {last}); \
         curve {curve:?}"
    );
    (first, last)
}

// ─────────────────────────────────────────────────────────────────────────
// Fixture construction — duplicated from `gguf_quantized_gpu.rs` (itself
// duplicated from `tests/it/gguf_qlora.rs`). Geometry, seeds, and amplitude
// are IDENTICAL so this file's measured cosines are directly comparable to
// both siblings' own baselines.
// ─────────────────────────────────────────────────────────────────────────

/// FNV-1a over `name`'s bytes (family J: deterministic, no unseeded RNG).
fn name_seed(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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

/// Workspace root — two levels up from this crate's manifest dir
/// (`crates/jammi-ai` -> workspace root); `CARGO_MANIFEST_DIR` is
/// crate-relative, not file-relative, so this is correct regardless of which
/// test binary in this crate evaluates it.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn cookbook_fixture(name: &str) -> PathBuf {
    workspace_root()
        .join("cookbook")
        .join("fixtures")
        .join(name)
}

fn fixture_url(name: &str) -> String {
    format!(
        "file://{}",
        workspace_root()
            .join("tests")
            .join("fixtures")
            .join(name)
            .display()
    )
}

fn write_tokenizer(dir: &Path) {
    std::fs::copy(
        cookbook_fixture("tiny_bert").join("tokenizer.json"),
        dir.join("tokenizer.json"),
    )
    .unwrap();
}

fn write_f32_checkpoint(dir: &Path, tensors: &HashMap<String, Tensor>) {
    std::fs::create_dir_all(dir).unwrap();
    candle_core::safetensors::save(tensors, dir.join("model.safetensors")).unwrap();
}

/// Mirrors `gguf_quantized_gpu.rs::write_gguf_checkpoint`: every matmul-site
/// `.weight` tensor is quantized at `quant`; every other tensor is written as
/// an `F32`-"quantized" `QTensor` (GGUF's lossless dense convention).
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

fn write_q8_0_gguf_fixture(dir: &Path) {
    let device = Device::Cpu;
    let tensors = bert_tensor_map(&device);
    let sites = bert_matmul_site_prefixes();
    write_json(dir, "config.json", &bert_config_json());
    write_tokenizer(dir);
    write_gguf_checkpoint(dir, &tensors, &sites, GgmlDType::Q8_0);
}

fn write_f32_reference_fixture(dir: &Path) {
    let device = Device::Cpu;
    let tensors = bert_tensor_map(&device);
    write_json(dir, "config.json", &bert_config_json());
    write_tokenizer(dir);
    write_f32_checkpoint(dir, &tensors);
}

fn local_id(dir: &Path) -> String {
    format!("local:{}", dir.display())
}

/// The five-sentence text set the CUDA and CPU siblings both embed, reused
/// here so this file's measured cosines are directly comparable.
const TEXTS: [&str; 5] = [
    "the quick brown fox",
    "jumps over the lazy dog",
    "hello world",
    "gguf quantized inference test",
    "a b c d e f g",
];

// ─────────────────────────────────────────────────────────────────────────
// Oracle (1): GGUF embed parity CPU<->Metal, over the SAME quantized
// checkpoint. See this file's module doc for why the floor is NOT Half A's
// CUDA Q8_1-activation-quantization number.
// ─────────────────────────────────────────────────────────────────────────

/// Measured on this Mac's Metal device, 2026-08-31: worst-case cosine across
/// the five-sentence `TEXTS` set was `>= 0.999999` (see the test's own
/// `eprintln!` for the exact run's number — printed every run so a future
/// regression is visible even though this floor itself does not move on
/// every run). Pinned with real headroom under that measurement rather than
/// at it, per this file's own "measured, not transcribed" doctrine (family
/// F) — 0.999 leaves five nines of margin below the observed six-nines
/// result while still catching a real kernel/dtype bug (which collapses
/// cosine far below 0.99, per every sibling suite's own documented claim).
const GGUF_METAL_EMBED_COSINE_FLOOR: f64 = 0.999;

/// A companion elementwise absolute-tolerance backstop. The fixture's own
/// known weight amplitude (`det_vec`'s `* 0.1` scale) bounds any single-lane
/// blowup; `5e-3` is generous headroom over the sub-`1e-5` deltas actually
/// observed on-device (see the test's own printed `worst_abs`).
const GGUF_METAL_ELEMENTWISE_ABS_TOL: f64 = 5e-3;

#[tokio::test(flavor = "multi_thread")]
async fn gguf_embedding_cpu_metal_parity() {
    skip_without_gpu!();
    loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    let model = local_id(&gguf_dir);

    let cpu_dir = TempDir::new().unwrap();
    let cpu = cpu_session(cpu_dir.path()).await;
    let gpu_dir = TempDir::new().unwrap();
    let gpu = gpu_session(gpu_dir.path()).await;

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
            "CPU and Metal GGUF query vectors must share a dimension"
        );
        for &v in cpu_v.iter().chain(gpu_v.iter()) {
            total_values += 1;
            if v.is_finite() {
                finite_values += 1;
            }
        }
        let cos = cosine(&cpu_v, &gpu_v);
        let abs = max_abs_diff(&cpu_v, &gpu_v);
        tracing::info!(text, cos, abs, "GGUF CPU<->Metal embed parity");
        worst_cos = worst_cos.min(cos);
        worst_abs = worst_abs.max(abs);
    }

    // F9: every value finite BY COUNT, never a vacuous "some finite" pass.
    assert_eq!(
        finite_values, total_values,
        "expected every GGUF CPU/Metal embedding value finite, got {finite_values}/{total_values}"
    );

    eprintln!(
        "gguf_embedding_cpu_metal_parity: worst_cos={worst_cos} floor={GGUF_METAL_EMBED_COSINE_FLOOR} \
         worst_abs={worst_abs} elementwise_tol={GGUF_METAL_ELEMENTWISE_ABS_TOL}"
    );
    assert!(
        worst_cos >= GGUF_METAL_EMBED_COSINE_FLOOR,
        "GGUF CPU<->Metal worst-case cosine {worst_cos} below floor {GGUF_METAL_EMBED_COSINE_FLOOR} \
         — a real kernel/dtype bug, not ordinary reduction-order noise"
    );
    assert!(
        worst_abs <= GGUF_METAL_ELEMENTWISE_ABS_TOL,
        "GGUF CPU<->Metal worst-case |delta| {worst_abs} exceeds {GGUF_METAL_ELEMENTWISE_ABS_TOL} \
         — a real kernel/dtype bug, not ordinary reduction-order noise"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (2): GGUF-on-Metal vs f32-on-Metal, same underlying weights — the
// quantization-LOSS floor (Q8_0 weight rounding only; both arms run on the
// same device, so no cross-device mechanism enters this comparison).
// ─────────────────────────────────────────────────────────────────────────

/// `tests/it/gguf_qlora.rs`'s CPU-hermetic measurement (`min_cosine >=
/// 0.9999995` for this same fixture/text-set) is the device-independent
/// proof that Q8_0 weight-quantization loss itself is tiny; this test
/// reproduces the comparison with both arms on Metal instead of CPU. Pinned
/// at the SAME `0.999` floor Half A pins for its CUDA arm, for the same
/// reason: this is a bug-catching floor (a wrong dequantize path, wrong
/// dtype), not a re-derivation of the loss bound — the measured on-device
/// number is printed for the record.
const GGUF_VS_F32_METAL_COSINE_FLOOR: f64 = 0.999;

#[tokio::test(flavor = "multi_thread")]
async fn gguf_on_metal_vs_f32_on_metal_quantization_loss_floor() {
    skip_without_gpu!();
    loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    let f32_dir = tmp.path().join("f32_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    write_f32_reference_fixture(&f32_dir);

    let gguf_session_dir = TempDir::new().unwrap();
    let gguf_gpu = gpu_session(gguf_session_dir.path()).await;
    let f32_session_dir = TempDir::new().unwrap();
    let f32_gpu = gpu_session(f32_session_dir.path()).await;

    let gguf_model = local_id(&gguf_dir);
    let f32_model = local_id(&f32_dir);

    let mut worst_cos = 1.0f64;
    let mut mean_cos = 0.0f64;
    for text in TEXTS {
        let gguf_v = gguf_gpu.encode_text_query(&gguf_model, text).await.unwrap();
        let f32_v = f32_gpu.encode_text_query(&f32_model, text).await.unwrap();
        assert_eq!(gguf_v.len(), f32_v.len(), "must share a dimension");
        let cos = cosine(&gguf_v, &f32_v);
        tracing::info!(
            text,
            cos,
            "GGUF-on-Metal vs f32-on-Metal quantization-loss cosine"
        );
        worst_cos = worst_cos.min(cos);
        mean_cos += cos / TEXTS.len() as f64;
    }

    eprintln!(
        "gguf_on_metal_vs_f32_on_metal: worst_cos={worst_cos} mean_cos={mean_cos} \
         floor={GGUF_VS_F32_METAL_COSINE_FLOOR}"
    );
    assert!(
        worst_cos > GGUF_VS_F32_METAL_COSINE_FLOOR,
        "GGUF-on-Metal vs f32-on-Metal worst-case cosine {worst_cos} at or below the pinned \
         floor {GGUF_VS_F32_METAL_COSINE_FLOOR} — either the Metal dequantize/dtype path is \
         broken, or the CPU-measured floor no longer holds on device"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (3): QLoRA-on-Metal smoke, GGUF base — mirrors Half A's Oracle 3
// exactly, with a Metal device instead of CUDA.
// ─────────────────────────────────────────────────────────────────────────

async fn add_training_source(session: &Arc<InferenceSession>) {
    session
        .add_source(
            "training",
            SourceType::File,
            SourceConnection {
                url: Some(fixture_url("training_pairs.csv")),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn qlora_learns_on_metal_with_gguf_base() {
    skip_without_gpu!();
    loss_capture::install();
    loss_capture::reset();

    let fixture_dir = TempDir::new().unwrap();
    let gguf_dir = fixture_dir.path().join("gguf_base");
    write_q8_0_gguf_fixture(&gguf_dir);
    let model = local_id(&gguf_dir);

    let dir = TempDir::new().unwrap();
    let session = gpu_session(dir.path()).await;
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

    // (a) completes on Metal, over a Quantized (GGUF) frozen base.
    job.wait().await.unwrap();
    let record = session
        .catalog()
        .get_training_job(&job.job_id)
        .await
        .unwrap();
    assert_eq!(
        record.status, "completed",
        "Metal QLoRA job should complete, got {}",
        record.status
    );

    // (b) loss decreases first->last epoch.
    let curve = loss_capture::captured();
    let (first, last) = assert_loss_decreases("qlora_gguf_metal", &curve);

    // (c) the adapter changes embeddings vs the (quantized) base model.
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
        "Metal QLoRA-trained adapter must change embeddings (LoRA delta non-zero), delta={delta}"
    );

    tracing::info!(
        first_loss = first,
        last_loss = last,
        embed_delta = delta,
        "QLoRA learns on Metal over a GGUF base"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Oracle (4): throughput baseline — printed only, no assertion. Perf
// assertions belong to the perf-claims machinery, not this correctness
// suite.
// ─────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn gguf_vs_f32_metal_throughput_baseline() {
    skip_without_gpu!();
    loss_capture::install();

    let tmp = TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf_model");
    let f32_dir = tmp.path().join("f32_model");
    write_q8_0_gguf_fixture(&gguf_dir);
    write_f32_reference_fixture(&f32_dir);

    let gguf_session_dir = TempDir::new().unwrap();
    let gguf_gpu = gpu_session(gguf_session_dir.path()).await;
    let f32_session_dir = TempDir::new().unwrap();
    let f32_gpu = gpu_session(f32_session_dir.path()).await;

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
        "gguf_vs_f32_metal_throughput_baseline: rows={ROWS} \
         gguf_rows_per_sec={gguf_rows_per_sec:.2} (elapsed={gguf_elapsed:?}) \
         f32_rows_per_sec={f32_rows_per_sec:.2} (elapsed={f32_elapsed:?}) \
         -- printed baseline only, no assertion (perf-claims machinery owns thresholds)"
    );
}
