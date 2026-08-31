//! Issue #351 (GGUF/k-quant inference + fail-loud QLoRA), phase-6 acceptance
//! suite. Every fixture below is built PROGRAMMATICALLY (no checked-in
//! binary `.gguf`/`.safetensors` file): a tiny BERT-shaped checkpoint's
//! tensors are generated deterministically, written once as an ordinary
//! `model.safetensors` (the F32 reference) and once as a `model.gguf` (via
//! `candle_core::quantized::gguf_file::write`, quantizing the matmul-site
//! tensors and densifying everything else at `F32` — exactly the shape
//! `QTensor::quantize` + `gguf_file::write` the design pin calls for).
//!
//! These tests do not exist on the pre-#351 tree (the RED oracle the
//! design's phase-6 gate names): `model::WeightsFormat::Gguf`,
//! `LoadedModel::quantization`, and the `model.gguf` resolver/backend arms
//! this file exercises are all new in this unit.

use std::collections::HashMap;
use std::path::Path;

use arrow::array::{ArrayRef, StringArray};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{BackendType, LoadedModel, ModelSource, ModelTask, WeightsFormat};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_db::store::manifest::{
    ComputeDevice, DefinitionHash, MaterializationEnv, MaterializationManifest, ModelIdentity,
    ProducingDescriptor,
};
use jammi_lora::{FrozenBase, LoraInitMode, LoraLinear, QuantizedLinear};
use jammi_numerics::WeightQuantization;
use std::sync::Arc;
use tempfile::tempdir;

// ─────────────────────────────────────────────────────────────────────────
// Deterministic fixture construction (family J: no unseeded RNG anywhere)
// ─────────────────────────────────────────────────────────────────────────

/// FNV-1a over `name`'s bytes — a stable per-tensor-name seed so every
/// tensor in the fixture gets its OWN deterministic value stream without a
/// hand-maintained counter.
fn name_seed(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `n` deterministic small-magnitude values, keyed by `name` — fixed fold
/// order, no RNG (family J).
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

/// A raw (no `"bert."` wrapper) BERT-family tensor map: every tensor
/// `jammi_encoders::bert::Bert::builder().build()` reads, by fully-qualified
/// name — the same names [`bert_matmul_site_prefixes`] below names as
/// matmul sites, plus embeddings/LayerNorms.
fn bert_tensor_map(
    hidden: usize,
    layers: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    type_vocab: usize,
    device: &Device,
) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    let add = |map: &mut HashMap<String, Tensor>, name: String, dims: &[usize]| {
        let t = det_tensor(&name, dims, device);
        map.insert(name, t);
    };
    add(
        &mut map,
        "embeddings.word_embeddings.weight".into(),
        &[vocab, hidden],
    );
    add(
        &mut map,
        "embeddings.position_embeddings.weight".into(),
        &[max_pos, hidden],
    );
    add(
        &mut map,
        "embeddings.token_type_embeddings.weight".into(),
        &[type_vocab, hidden],
    );
    add(&mut map, "embeddings.LayerNorm.weight".into(), &[hidden]);
    add(&mut map, "embeddings.LayerNorm.bias".into(), &[hidden]);
    for n in 0..layers {
        let p = format!("encoder.layer.{n}");
        for site in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ] {
            add(&mut map, format!("{p}.{site}.weight"), &[hidden, hidden]);
            add(&mut map, format!("{p}.{site}.bias"), &[hidden]);
        }
        add(
            &mut map,
            format!("{p}.intermediate.dense.weight"),
            &[intermediate, hidden],
        );
        add(
            &mut map,
            format!("{p}.intermediate.dense.bias"),
            &[intermediate],
        );
        add(
            &mut map,
            format!("{p}.output.dense.weight"),
            &[hidden, intermediate],
        );
        add(&mut map, format!("{p}.output.dense.bias"), &[hidden]);
        for ln in ["attention.output.LayerNorm", "output.LayerNorm"] {
            add(&mut map, format!("{p}.{ln}.weight"), &[hidden]);
            add(&mut map, format!("{p}.{ln}.bias"), &[hidden]);
        }
    }
    map
}

/// The six per-layer matmul-site tensor-name PREFIXES (no `.weight`/`.bias`
/// suffix) for a raw (unwrapped) BERT checkpoint — mirrors
/// `jammi_ai::model::backend::gguf::matmul_site_names`'s own `Bert` arm
/// (that module is `pub(crate)` inside `jammi-ai`, unreachable from this
/// external integration-test crate, so this is an independent re-derivation
/// from `jammi_encoders::bert`'s own per-layer module names — the SAME
/// names `LoraSite::resolve_base` uses, read directly off that source).
fn bert_matmul_site_prefixes(layers: usize) -> Vec<String> {
    let mut v = Vec::new();
    for n in 0..layers {
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

fn bert_config_json(
    hidden: usize,
    layers: usize,
    heads: usize,
    intermediate: usize,
    vocab: usize,
    max_pos: usize,
    type_vocab: usize,
) -> serde_json::Value {
    serde_json::json!({
        "model_type": "bert",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": max_pos,
        "type_vocab_size": type_vocab,
        "layer_norm_eps": 1e-12,
    })
}

fn write_json(dir: &Path, name: &str, value: &serde_json::Value) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(name), serde_json::to_string(value).unwrap()).unwrap();
}

fn write_tokenizer(dir: &Path) {
    std::fs::copy(
        jammi_test_utils::cookbook_fixture("tiny_bert").join("tokenizer.json"),
        dir.join("tokenizer.json"),
    )
    .unwrap();
}

fn write_f32_checkpoint(dir: &Path, tensors: &HashMap<String, Tensor>) {
    std::fs::create_dir_all(dir).unwrap();
    candle_core::safetensors::save(tensors, dir.join("model.safetensors")).unwrap();
}

/// Write `dir/model.gguf`: every tensor named in `matmul_sites` (its
/// `.weight` form) is quantized at `quant`; every OTHER tensor (embeddings,
/// LayerNorms, biases) is written as an `F32`-"quantized" `QTensor` — GGUF's
/// own convention for a dense-stored tensor (`QTensor::quantize(t,
/// GgmlDType::F32)` is a legitimate, lossless wrap: `F32`'s block size is 1).
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

fn embed(model: &LoadedModel, text: &str) -> Vec<f32> {
    let content: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text])) as ArrayRef];
    let output = model.forward(&content, ModelTask::TextEmbedding).unwrap();
    output.float_outputs[0].clone()
}

fn device_config() -> DeviceConfig {
    DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    }
}

async fn try_resolve(
    dir: &Path,
    backend_hint: Option<BackendType>,
) -> jammi_db::error::Result<jammi_ai::model::ResolvedModel> {
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let source = ModelSource::local(dir);
    resolver
        .resolve(&source, ModelTask::TextEmbedding, backend_hint)
        .await
}

async fn resolve_and_load(dir: &Path) -> LoadedModel {
    let resolved = try_resolve(dir, None).await.unwrap();
    let backend = CandleBackend;
    backend.load(&resolved, &device_config()).unwrap()
}

async fn try_resolve_and_load(
    dir: &Path,
    backend_hint: Option<BackendType>,
) -> jammi_db::error::Result<LoadedModel> {
    let resolved = try_resolve(dir, backend_hint).await?;
    let backend = CandleBackend;
    backend.load(&resolved, &device_config())
}

/// The standard tiny fixture geometry every A1/A2/A4 test below shares:
/// `hidden=32, layers=1, heads=2, intermediate=128` — small enough to run
/// fast, `hidden % 32 == 0` so `q8_0`/`q4_0` quantize without a block-size
/// refusal (both need `last_dim % 32 == 0`).
const HIDDEN: usize = 32;
const LAYERS: usize = 1;
const HEADS: usize = 2;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 256;
const MAX_POS: usize = 128;
const TYPE_VOCAB: usize = 2;

fn small_fixture(device: &Device) -> (HashMap<String, Tensor>, serde_json::Value, Vec<String>) {
    let tensors = bert_tensor_map(
        HIDDEN,
        LAYERS,
        INTERMEDIATE,
        VOCAB,
        MAX_POS,
        TYPE_VOCAB,
        device,
    );
    let config = bert_config_json(
        HIDDEN,
        LAYERS,
        HEADS,
        INTERMEDIATE,
        VOCAB,
        MAX_POS,
        TYPE_VOCAB,
    );
    let sites = bert_matmul_site_prefixes(LAYERS);
    (tensors, config, sites)
}

// ─────────────────────────────────────────────────────────────────────────
// A1: resolve -> load -> embed through the public embedding path (CPU);
// cosine vs. the f32 safetensors run of the SAME checkpoint; all values
// finite by COUNT.
// ─────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn gguf_embedding_matches_f32_reference_within_a_measured_cosine_floor() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);

    let tmp = tempdir().unwrap();
    let f32_dir = tmp.path().join("f32_model");
    let gguf_dir = tmp.path().join("gguf_model");
    for dir in [&f32_dir, &gguf_dir] {
        write_json(dir, "config.json", &config);
        write_tokenizer(dir);
    }
    write_f32_checkpoint(&f32_dir, &tensors);
    write_gguf_checkpoint(&gguf_dir, &tensors, &sites, GgmlDType::Q8_0);

    let f32_model = resolve_and_load(&f32_dir).await;
    let gguf_model = resolve_and_load(&gguf_dir).await;

    let texts = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "hello world",
        "gguf quantized inference test",
        "a b c d e f g",
    ];

    let mut total_values = 0usize;
    let mut finite_values = 0usize;
    let mut cosines = Vec::with_capacity(texts.len());
    for text in texts {
        let a = embed(&f32_model, text);
        let b = embed(&gguf_model, text);
        assert_eq!(a.len(), b.len());
        for &v in a.iter().chain(b.iter()) {
            total_values += 1;
            if v.is_finite() {
                finite_values += 1;
            }
        }
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        cosines.push(dot / (na * nb));
    }

    // F9: every embedding value must be finite, BY COUNT (never a "some
    // finite" vacuous pass).
    assert_eq!(
        finite_values, total_values,
        "expected every embedding value finite, got {finite_values}/{total_values}"
    );

    let mean_cosine: f32 = cosines.iter().sum::<f32>() / cosines.len() as f32;
    let min_cosine: f32 = cosines.iter().cloned().fold(f32::INFINITY, f32::min);
    // MEASURED (F9), not assumed: on this workspace's hermetic CPU dev/CI
    // arm (2026-08-30), a `q8_0`-quantized 1-layer/32-dim BERT tower's
    // pooled embedding vs. its F32 reference (identical underlying weight
    // values) measured `mean_cosine=0.99999964`, `min_cosine=0.9999995`
    // over the five-sentence fixture set above — `q8_0` is the
    // LOWEST-error k-quant format this workspace supports (one `f16`
    // scale per 32-element block), so this near-1.0 result is expected,
    // not suspicious. The floor below (0.999) sits with a wide margin
    // under the measured value while still being a MEANINGFUL bound (a
    // broken dequantize/dtype-cast path would land far below it), not a
    // threshold tuned to merely-pass.
    assert!(
        mean_cosine > 0.999,
        "mean cosine {mean_cosine} (min {min_cosine}) below the measured floor; cosines={cosines:?}"
    );
    // MEASURED (F9), not assumed — re-confirmed on this workspace's hermetic
    // CPU dev/CI arm (2026-08-31, issue #351 wave 5 audit): `min_cosine=
    // 0.9999995`, matching the doc comment above exactly. `min_cosine` was
    // previously computed and printed into failure messages but never
    // itself asserted — a per-sentence outlier well below the mean could
    // slip through unnoticed. The floor here (0.999, the SAME wide-margin
    // floor `mean_cosine` already clears) is a real, non-vacuous bound: a
    // broken dequantize/dtype-cast path corrupting even ONE sentence's
    // embedding would land its cosine far below this, not merely nudge the
    // mean.
    assert!(
        min_cosine > 0.999,
        "min cosine {min_cosine} (mean {mean_cosine}) below the measured floor; cosines={cosines:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// A2: ModelIdentity.quantization is Some(...); definition_hash differs from
// the f32 run's; digest byte-mutation changes content_digest; in-place
// mutation marks a warm ModelCache entry stale.
// ─────────────────────────────────────────────────────────────────────────

fn definition_hash_for(model_id: &str, model: &LoadedModel) -> DefinitionHash {
    let identity = ModelIdentity {
        model_id: model_id.to_string(),
        backend: model.backend_kind().to_string(),
        compute_precision: model.compute_precision(),
        content_digest: model.content_digest().unwrap(),
        quantization: model.quantization(),
    };
    let descriptor = ProducingDescriptor::Embedding {
        model_id: model_id.to_string(),
        task: ModelTask::TextEmbedding,
        source_id: "gguf_qlora_source".to_string(),
        columns: vec!["text".to_string()],
        key_column: "id".to_string(),
        dimensions: HIDDEN,
    };
    let env = MaterializationEnv::new(ComputeDevice::Cpu, vec![identity]);
    MaterializationManifest::definition_of(&descriptor, &env).unwrap()
}

#[tokio::test]
async fn gguf_model_identity_reports_quantization_and_a_distinct_definition_hash() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);

    let tmp = tempdir().unwrap();
    let f32_dir = tmp.path().join("f32_model");
    let gguf_dir = tmp.path().join("gguf_model");
    for dir in [&f32_dir, &gguf_dir] {
        write_json(dir, "config.json", &config);
        write_tokenizer(dir);
    }
    write_f32_checkpoint(&f32_dir, &tensors);
    write_gguf_checkpoint(&gguf_dir, &tensors, &sites, GgmlDType::Q8_0);

    let f32_model = resolve_and_load(&f32_dir).await;
    let gguf_model = resolve_and_load(&gguf_dir).await;

    assert_eq!(
        f32_model.quantization(),
        None,
        "a safetensors load must report no weight-quantization format"
    );
    assert_eq!(
        gguf_model.quantization(),
        Some(WeightQuantization::Q8_0),
        "every matmul-site tensor in this fixture was quantized at q8_0, so the \
         modal quantized dtype must be exactly q8_0"
    );

    let f32_hash = definition_hash_for(&ModelSource::local(&f32_dir).to_string(), &f32_model);
    let gguf_hash = definition_hash_for(&ModelSource::local(&gguf_dir).to_string(), &gguf_model);
    assert_ne!(
        f32_hash, gguf_hash,
        "a GGUF-quantized load and its f32 reference must never collide on one \
         DefinitionHash even under otherwise-identical descriptors"
    );
}

#[tokio::test]
async fn gguf_byte_mutation_changes_content_digest() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("gguf_model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let model_before = resolve_and_load(&dir).await;
    let digest_before = model_before.content_digest().unwrap();
    drop(model_before);

    let weights_path = dir.join("model.gguf");
    let mut bytes = std::fs::read(&weights_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&weights_path, &bytes).unwrap();

    let model_after = resolve_and_load(&dir).await;
    let digest_after = model_after.content_digest().unwrap();

    assert_ne!(
        digest_before, digest_after,
        "mutating model.gguf bytes in place, under a constant model_id, must \
         change the content digest"
    );
}

#[tokio::test]
async fn warm_cache_reload_after_gguf_in_place_mutation_reports_a_fresh_digest() {
    use jammi_ai::concurrency::GpuScheduler;
    use jammi_ai::model::cache::ModelCache;

    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("gguf_model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    let cache = ModelCache::new(resolver, device_config(), scheduler);
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let digest1 = guard1.model.content_digest().unwrap();
    drop(guard1);

    // In-place mutation, length-changing (never rests on sub-second mtime
    // resolution alone): append a byte.
    let weights_path = dir.join("model.gguf");
    let mut bytes = std::fs::read(&weights_path).unwrap();
    bytes.push(0xAB);
    std::fs::write(&weights_path, &bytes).unwrap();

    let guard_warm = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let digest_warm = guard_warm.model.content_digest().unwrap();

    assert_ne!(
        digest1, digest_warm,
        "a warm ModelCache hit after an in-place model.gguf mutation must \
         report a FRESH digest — the staleness fingerprint must mark the entry \
         stale and force a reload, never keep attesting the pre-mutation digest"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// A3 (QLoRA): per-site input-gradient parity between a `FrozenBase::Quantized`
// site and a `FrozenBase::Dense` (dequantized) reference, plus a
// loss-decrease smoke check on both.
// ─────────────────────────────────────────────────────────────────────────

fn build_lora(base: FrozenBase, varmap: &VarMap, device: &Device, seed: u64) -> LoraLinear {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, device).pp("site");
    let mut lora = LoraLinear::new_with_base(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        seed,
        varmap,
        &vb,
    )
    .unwrap();
    lora.set_training(false); // eager composition on both arms — no dropout, no fused-vs-eager asymmetry
    lora
}

#[test]
fn qlora_input_gradient_parity_vs_dense_dequantized_reference() {
    let device = Device::Cpu;
    let (out_f, in_f, rows) = (8usize, 32usize, 3usize);
    let seed = 42u64;

    let w_v = det_vec("qlora_parity_weight", out_f * in_f);
    let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
    let bias_v = det_vec("qlora_parity_bias", out_f);
    let bias = Tensor::from_vec(bias_v, out_f, &device).unwrap();

    let wq = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap());
    let w_deq = wq.dequantize(&device).unwrap();

    let quantized_base =
        FrozenBase::Quantized(QuantizedLinear::new(wq, Some(bias.clone())).unwrap());
    let dense_base = FrozenBase::Dense(Linear::new(w_deq, Some(bias)));

    let varmap_q = VarMap::new();
    let varmap_d = VarMap::new();
    let lora_q = build_lora(quantized_base, &varmap_q, &device, seed);
    let lora_d = build_lora(dense_base, &varmap_d, &device, seed);

    // Same seed, same prefix ("site"), same rank/alpha/init_mode -> A/B init
    // must be bit-identical across the two constructions.
    assert_eq!(
        lora_q
            .lora_a
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        lora_d
            .lora_a
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        "lora_a must be seed-identical across the Quantized and Dense constructions"
    );
    assert_eq!(
        lora_q
            .lora_b
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        lora_d
            .lora_b
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        "lora_b must be seed-identical across the Quantized and Dense constructions"
    );

    let x_v = det_vec("qlora_parity_input", rows * in_f);
    let x_q =
        Var::from_tensor(&Tensor::from_vec(x_v.clone(), (rows, in_f), &device).unwrap()).unwrap();
    let x_d = Var::from_tensor(&Tensor::from_vec(x_v, (rows, in_f), &device).unwrap()).unwrap();

    let y_q = lora_q.forward(x_q.as_tensor()).unwrap();
    let y_d = lora_d.forward(x_d.as_tensor()).unwrap();

    // Forward parity first (a prerequisite for gradient parity meaning
    // anything): the Quantized site's own q8_0 rounding error is the ONLY
    // source of divergence from the dense-dequantized reference — both run
    // the identical LoRA epilogue over the identical A/B tensors.
    let y_q_v = y_q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let y_d_v = y_d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let fwd_max_abs_diff = y_q_v
        .iter()
        .zip(&y_d_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    let loss_q = y_q.sum_all().unwrap();
    let loss_d = y_d.sum_all().unwrap();
    let grads_q = loss_q.backward().unwrap();
    let grads_d = loss_d.backward().unwrap();
    let grad_q = grads_q.get(x_q.as_tensor()).unwrap();
    let grad_d = grads_d.get(x_d.as_tensor()).unwrap();

    let grad_q_v = grad_q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let grad_d_v = grad_d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        grad_q_v.iter().all(|v| v.is_finite()) && grad_d_v.iter().all(|v| v.is_finite()),
        "both input gradients must be finite"
    );
    let grad_max_abs_diff = grad_q_v
        .iter()
        .zip(&grad_d_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    // MEASURED (F9), not assumed, on this workspace's hermetic CPU dev/CI
    // arm (2026-08-30), `out_f=8, in_f=32, rank=4`:
    // `fwd_max_abs_diff=1.9994e-4` (the q8_0 forward-rounding error — the
    // fused quantized-matmul kernel's own summation order differs slightly
    // from a plain dense matmul over the identical dequantized weight),
    // `grad_max_abs_diff=0` EXACTLY (not merely small) — `bwd` computes
    // `dL/dx = dL/dy @ dequantize(W)^T` via the SAME single full
    // dequantize-then-matmul this test's own dense-dequantized reference
    // computes, so the two are bit-identical, not merely close. The
    // tolerances below (1e-2 for BOTH) are a wide margin over the measured
    // forward diff and are trivially satisfied — not vacuously, though:
    // asserted against the measured `grad_max_abs_diff` reported above
    // rather than assumed, per F9.
    assert!(
        fwd_max_abs_diff < 1e-2,
        "forward max abs diff {fwd_max_abs_diff} exceeds tolerance"
    );
    assert!(
        grad_max_abs_diff < 1e-2,
        "input-gradient max abs diff {grad_max_abs_diff} exceeds tolerance"
    );

    // Smoke: a single SGD step on lora_a/lora_b, driven by the JUST-COMPUTED
    // gradients, must strictly decrease the loss for BOTH the Quantized-
    // backed and the Dense-backed site — proof the QLoRA wiring is actually
    // trainable end-to-end, not merely differentiable in isolation.
    let lr = 0.05f64;
    for (varmap, lora, x, loss_before) in [
        (&varmap_q, &lora_q, &x_q, loss_q.to_vec0::<f32>().unwrap()),
        (&varmap_d, &lora_d, &x_d, loss_d.to_vec0::<f32>().unwrap()),
    ] {
        let grads = if std::ptr::eq(lora, &lora_q) {
            &grads_q
        } else {
            &grads_d
        };
        let data = varmap.data().lock().unwrap();
        for name in ["site.lora_a", "site.lora_b"] {
            let var = data.get(name).unwrap_or_else(|| panic!("missing {name}"));
            let grad = grads.get(var.as_tensor()).unwrap();
            let updated = (var.as_tensor() - (grad * lr).unwrap()).unwrap();
            var.set(&updated).unwrap();
        }
        drop(data);
        let y_after = lora.forward(x.as_tensor()).unwrap();
        let loss_after = y_after.sum_all().unwrap().to_vec0::<f32>().unwrap();
        assert!(
            loss_after.is_finite() && loss_after < loss_before,
            "one SGD step must strictly decrease the loss: before={loss_before} after={loss_after}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Advisory fold (issue #351 wave 5 audit): a training-mode `Quantized`
// forward with `lora_dropout > 0`, so the esc-032/033 dropout-stream
// reservation actually fires on the `FrozenBase::Quantized` arm at least
// once. `LoraLinear::forward`'s own doc ("Dropout key reservation")
// documents that `DropoutMasks::next_key` is called EXACTLY ONCE per
// training forward, reserved UNIFORMLY regardless of base storage format —
// but every training-mode forward this it-suite drove before this fold used
// `lora_dropout == 0` (the A3 parity test above explicitly disables
// training: `lora.set_training(false)`), so a Quantized base's dropout
// reservation had never actually executed under test. Drives
// `jammi_lora::LoraLinear` directly (its own public API — no change to
// `jammi-lora` itself).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn quantized_base_training_forward_with_dropout_reserves_the_dropout_stream() {
    let device = Device::Cpu;
    let (out_f, in_f, rows) = (8usize, 32usize, 3usize);
    let seed = 7u64;

    let w_v = det_vec("dropout_quantized_weight", out_f * in_f);
    let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
    let wq = Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap());
    let quantized_base = FrozenBase::Quantized(QuantizedLinear::new(wq, None).unwrap());

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device).pp("site");
    let mut lora = LoraLinear::new_with_base(
        quantized_base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3), // lora_dropout > 0
        seed,
        &varmap,
        &vb,
    )
    .unwrap();
    // `new_with_base` already constructs with `training: true` (module doc),
    // set explicitly here so this test's premise is visible without reading
    // that source.
    lora.set_training(true);

    assert_eq!(
        lora.dropout_position().unwrap(),
        Some(0),
        "a dropout-configured layer must start at forward-count 0, before any training forward"
    );

    let x = Tensor::from_vec(
        det_vec("dropout_quantized_input", rows * in_f),
        (rows, in_f),
        &device,
    )
    .unwrap();
    let y = lora.forward(&x).unwrap();
    let y_v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        y_v.iter().all(|v| v.is_finite()),
        "a training-mode Quantized forward with dropout must produce finite output: {y_v:?}"
    );

    // The oracle: ONE training forward over a Quantized base with
    // `lora_dropout > 0` must advance the dropout-stream position by
    // EXACTLY one (`DropoutMasks::next_key` called once, per `forward`'s
    // own doc) — proof the reservation fired on this arm, not skipped
    // (which would leave the position at 0, esc-033's O(1)-resume
    // invariant silently broken for every Quantized-base QLoRA run) and
    // not double-drawn (which would advance it by two).
    assert_eq!(
        lora.dropout_position().unwrap(),
        Some(1),
        "one training-mode forward over a Quantized base with lora_dropout > 0 must advance the \
         dropout-stream position by exactly one"
    );

    // A second forward must advance it again, to 2 — the reservation fires
    // on EVERY training forward, not merely the first.
    let _ = lora.forward(&x).unwrap();
    assert_eq!(
        lora.dropout_position().unwrap(),
        Some(2),
        "a second training-mode forward must advance the dropout-stream position again"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// A4: typed-error suite.
// ─────────────────────────────────────────────────────────────────────────

fn model_err_message(err: &JammiError) -> String {
    err.to_string()
}

/// `Result::unwrap_err` requires `T: Debug`; neither `ResolvedModel` nor
/// `LoadedModel` implement it (both carry non-`Debug` candle/tensor
/// internals), so every typed-error assertion below goes through this
/// instead.
fn expect_err<T>(r: jammi_db::error::Result<T>) -> JammiError {
    match r {
        Ok(_) => panic!("expected a typed error, got Ok"),
        Err(e) => e,
    }
}

/// Non-`model.gguf` `*.gguf` file present, `model.gguf` absent: a typed
/// refusal naming the found file(s) and the `model.gguf` convention.
#[tokio::test]
async fn other_gguf_filename_without_the_canonical_name_is_a_typed_refusal() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);
    std::fs::rename(dir.join("model.gguf"), dir.join("weights.gguf")).unwrap();

    let err = expect_err(try_resolve(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains("weights.gguf") && msg.contains("model.gguf"),
        "expected a typed refusal naming 'weights.gguf' and the 'model.gguf' \
         convention, got: {msg}"
    );
}

/// Dual-format precedence pin (oracle advisory (b), issue #351 wave 12; test
/// honesty fold, issue #351 wave 13 audit advisory 3): a LOCAL directory
/// carrying BOTH `model.safetensors` and `model.gguf` resolves to
/// `WeightsFormat::Safetensors`, with the `model.gguf` sibling wholly
/// ignored (never consulted, never even opened) — the frozen precedence
/// `resolve_local` has always applied, now pinned so a future refactor of
/// the local path can't accidentally acquire the Hub-path defect (a
/// fallback keyed off download/read FAILURE rather than presence).
///
/// `model.gguf` is corrupted BEFORE `try_resolve` is even called (not
/// merely before `load`), so a resolve that peeked at the GGUF file's
/// bytes for ANY reason — not just to decide the format, but e.g. to probe
/// it as a fallback — would itself fail here, not just produce a
/// wrong-but-not-crashing load. The resulting embedding is asserted EQUAL
/// (not merely computed and discarded) to a same-tensors safetensors-ONLY
/// reference directory's embedding of the identical text: since the local
/// path never quantizes (plain F32 safetensors, deterministic CPU forward),
/// this is a bit-exact equality, not a cosine floor — any accidental read
/// of the corrupted `model.gguf` would produce a load failure or garbage
/// output, not a value that happens to match the reference bit-for-bit.
#[tokio::test]
async fn local_dir_with_both_safetensors_and_gguf_resolves_to_safetensors_and_ignores_gguf() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();

    // The safetensors-ONLY reference: same tensors, no gguf sibling at all.
    let ref_dir = tmp.path().join("reference_model");
    write_json(&ref_dir, "config.json", &config);
    write_tokenizer(&ref_dir);
    write_f32_checkpoint(&ref_dir, &tensors);

    // The dual-format directory under test.
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    write_f32_checkpoint(&dir, &tensors);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    // Corrupt the (should-be-ignored) GGUF sibling BEFORE resolving at all —
    // proves the resolve decision never reads model.gguf's bytes, not merely
    // that a subsequent load doesn't.
    std::fs::write(dir.join("model.gguf"), b"not a real gguf file").unwrap();

    let resolved = try_resolve(&dir, None).await.unwrap();
    assert_eq!(
        resolved.weights_format,
        WeightsFormat::Safetensors,
        "a directory carrying both formats must resolve to safetensors, \
         with model.gguf ignored"
    );
    assert_eq!(
        resolved.weights_paths,
        vec![dir.join("model.safetensors")],
        "the resolved weights path must be model.safetensors, not model.gguf"
    );

    let backend = CandleBackend;
    let loaded = backend.load(&resolved, &device_config()).unwrap();
    let embedding = embed(&loaded, "dual-format precedence");

    let reference = resolve_and_load(&ref_dir).await;
    let reference_embedding = embed(&reference, "dual-format precedence");

    assert_eq!(
        embedding, reference_embedding,
        "the dual-format directory's embedding must EQUAL the safetensors-only \
         reference's embedding of the same text — any deviation would mean \
         model.gguf's (corrupted) bytes leaked into the loaded weights"
    );
}

/// `model.gguf` present, `config.json` absent: config.json stays REQUIRED
/// for the quantized arm, exactly as it is today for safetensors.
#[tokio::test]
async fn gguf_without_config_json_is_the_existing_missing_config_refusal() {
    let device = Device::Cpu;
    let (tensors, _config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    std::fs::create_dir_all(&dir).unwrap();
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let err = expect_err(try_resolve(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains("config.json"),
        "expected the existing missing-config refusal, got: {msg}"
    );
}

/// An unsupported GGML dtype (`q8_1` — a valid candle-core dtype, but named
/// by neither `WeightQuantization` nor the dense F32/F16/BF16 set) is a
/// typed refusal naming the tensor and dtype.
#[tokio::test]
async fn unsupported_ggml_dtype_is_a_typed_refusal_naming_tensor_and_dtype() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);

    // Hand-build the GGUF file so exactly one tensor uses Q8_1.
    std::fs::create_dir_all(&dir).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let target = format!("{}.weight", sites[0]);
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
    for name in names {
        let t = &tensors[name];
        let dtype = if *name == target {
            GgmlDType::Q8_1
        } else if sites.iter().any(|p| *name == format!("{p}.weight")) {
            GgmlDType::Q4_0
        } else {
            GgmlDType::F32
        };
        qtensors.push((name.clone(), QTensor::quantize(t, dtype).unwrap()));
    }
    let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    gguf_file::write(&mut writer, &[], &refs).unwrap();

    let err = expect_err(try_resolve_and_load(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains(&target) && msg.to_lowercase().contains("dtype"),
        "expected a typed refusal naming '{target}' and its unsupported dtype, got: {msg}"
    );
}

/// A required matmul-site tensor missing from the GGUF file: a typed
/// refusal LISTING the missing key.
#[tokio::test]
async fn missing_matmul_site_tensor_is_a_typed_refusal_listing_the_key() {
    let device = Device::Cpu;
    let (mut tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);

    let missing_key = format!("{}.weight", sites[1]);
    tensors.remove(&missing_key);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let err = expect_err(try_resolve_and_load(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains(&missing_key),
        "expected a typed refusal listing the missing tensor '{missing_key}', got: {msg}"
    );
}

/// A GGUF resolve for a HF-CLAP-typed config: "quantized serving not
/// supported for this architecture" (K2, never a silent degradation).
#[tokio::test]
async fn unsupported_architecture_gguf_is_a_typed_refusal() {
    let device = Device::Cpu;
    let (tensors, _bert_config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    let clap_config = serde_json::json!({
        "model_type": "clap_audio_model",
        "hidden_size": HIDDEN,
        "depths": [1],
        "num_attention_heads": [HEADS],
        "projection_dim": HIDDEN,
    });
    write_json(&dir, "config.json", &clap_config);
    // The refusal fires before any tensor data is read, so an empty tensor
    // map (a structurally-valid, empty GGUF file) is sufficient.
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let err = expect_err(try_resolve_and_load(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains("not supported for this architecture"),
        "expected the architecture refusal, got: {msg}"
    );
}

/// `backend_hint = Some(Ort)` against a directory that carries only
/// `model.gguf`: GGUF is Candle-only, so this is the SAME typed refusal an
/// ONNX-missing directory produces today.
#[tokio::test]
async fn ort_backend_with_only_gguf_present_is_a_typed_refusal() {
    let device = Device::Cpu;
    let (tensors, config, sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4_0);

    let err = expect_err(try_resolve(&dir, Some(BackendType::Ort)).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains("ONNX"),
        "expected the existing 'No ONNX weights found for ORT backend' refusal, got: {msg}"
    );
}

/// A corrupt/truncated `model.gguf` header: a typed RESOLVER refusal at
/// resolve time (header parse failure), never a silent fallback to raw
/// file-size estimation.
#[tokio::test]
async fn corrupt_gguf_header_is_a_typed_resolver_refusal() {
    let device = Device::Cpu;
    let (_tensors, config, _sites) = small_fixture(&device);
    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    std::fs::write(
        dir.join("model.gguf"),
        b"not a real gguf file, just garbage bytes",
    )
    .unwrap();

    let err = expect_err(try_resolve(&dir, None).await);
    let msg = model_err_message(&err);
    assert!(
        msg.contains("GGUF") || msg.to_lowercase().contains("magic"),
        "expected a typed header-parse refusal, got: {msg}"
    );
}

/// A `q4k` case over a 256-dim tower (`q4k`'s block size is 256, so this is
/// the one format this fixture set cannot exercise at `hidden=32`).
#[tokio::test]
async fn q4k_over_a_256_dim_tower_resolves_and_loads_successfully() {
    let device = Device::Cpu;
    let (hidden, layers, heads, intermediate, vocab, max_pos, type_vocab) = (
        256usize, 1usize, 4usize, 256usize, 256usize, 128usize, 2usize,
    );
    let tensors = bert_tensor_map(
        hidden,
        layers,
        intermediate,
        vocab,
        max_pos,
        type_vocab,
        &device,
    );
    let config = bert_config_json(
        hidden,
        layers,
        heads,
        intermediate,
        vocab,
        max_pos,
        type_vocab,
    );
    let sites = bert_matmul_site_prefixes(layers);

    let tmp = tempdir().unwrap();
    let dir = tmp.path().join("model");
    write_json(&dir, "config.json", &config);
    write_tokenizer(&dir);
    write_gguf_checkpoint(&dir, &tensors, &sites, GgmlDType::Q4K);

    let model = resolve_and_load(&dir).await;
    assert_eq!(model.quantization(), Some(WeightQuantization::Q4K));
    let v = embed(&model, "the quick brown fox");
    assert!(
        v.iter().all(|x| x.is_finite()),
        "embedding must be finite: {v:?}"
    );
}
