//! The identity-audited encode-step tier: rows in a table → persisted
//! embeddings, through the engine's real text-embedding serving surface —
//! [`generate_text_embeddings`](jammi_ai::session::InferenceSession::generate_text_embeddings),
//! the SAME `resolve -> tokenize -> forward -> pool -> normalize -> write` path
//! a serving request walks — swept over row counts, folded into a
//! [`crate::report::EncodeStepTier`] whose declared `IDENTITY_FIELDS` name the
//! COMPLETE output-affecting parameter set for this surface. See
//! [`crate::report::EncodeStepTier`]'s own doc for the identity-completeness
//! rationale this tier protects at the bench-comparison layer.
//!
//! ## What one run answers
//!
//! * **How fast, how small** — per row count: the warm serve's p50 and fastest
//!   wall time, rows/s and real-token/s at the p50, the peak host RSS and the
//!   peak device-memory growth; and, reported apart, what the FIRST call cost
//!   (the model load, then the first serve on a session nothing has warmed).
//! * **Where the cost sits** — the sweep is fitted to
//!   [`crate::timing::CostFit`], `serve_ms = fixed_ms + per_row_ms · rows`, so
//!   the per-call cost of the plan and the per-row cost of the work are two
//!   numbers, at whatever `[inference] partitions` the run was given.
//! * **Against what** — the run can leave the corpus it served and the vectors
//!   it persisted in an exchange directory, which
//!   `reference/torch_encode.py` reads to do the same work in PyTorch and to
//!   check, row by row, that the two stacks embedded the same thing.
//!
//! ## One process per sweep point
//!
//! The kernel's resident-set high-water mark (`VmHWM`) never falls, so a sweep
//! in one process could attribute only its largest point's memory, and that
//! only if it ran last. A run over several row counts therefore measures each
//! in a fresh child of this binary — the same subcommand, given the one row
//! count — and folds the children's points into one tier; a run over one row
//! count measures in-process. A child's cold first call is also honestly
//! cold: no earlier point loaded the model into it.
//!
//! ## Not a synthetic loop
//!
//! Every number on the emitted tier is either read off a real artifact or
//! produced by a real call into the engine:
//!
//! * `checkpoint_config_sha256`/`checkpoint_weights_sha256`/
//!   `checkpoint_weights_size_bytes`/`checkpoint_tokenizer_sha256` are
//!   `sha256_and_len` over the model dir's actual bytes (the SAME helper
//!   `finetune_step.rs`/`grad_oracle.rs` use) — the complete three-file
//!   checkpoint content identity, never a two-file subset (tokenizer bytes are
//!   output-affecting on this surface, see
//!   [`crate::report::EncodeStepTier::checkpoint_tokenizer_sha256`]'s doc).
//! * `compute_precision`, `pooling` and `max_sequence_length` are read off the
//!   LOADED model ([`jammi_ai::model::LoadedModel::compute_precision`],
//!   [`resolved_pooling`](jammi_ai::model::LoadedModel::resolved_pooling),
//!   [`max_sequence_length`](jammi_ai::model::LoadedModel::max_sequence_length),
//!   via the tier's own session model cache) — what the serve actually ran
//!   with, never a constant mirroring a fixture or a value re-derived from
//!   `config.json`. `checkpoint_pooling_sha256` closes the companion gap: the
//!   pooling-CONFIG BYTES themselves, hashed with the identical presence gate
//!   the engine's own `content_digest` applies to `1_Pooling/config.json`.
//! * `device_requested` is the CLI/param device value declared BEFORE any
//!   compute runs; `device_name` is the post-hoc hardware fact only knowable
//!   after the device resolved — see [`crate::report::EncodeStepTier`]'s doc
//!   for the identity-vs-provenance split.
//! * Each point's token counts are a REAL tokenization of the corpus text
//!   through [`jammi_ai::model::tokenizer::TokenizerWrapper`] — the exact
//!   wrapper `CandleBackend` loads for a local model — over the model's own
//!   `tokenizer.json`, truncated at the loaded model's bound, in the
//!   `batch_size` chunks the plan forwards.
//! * The measured serve is `crate::model_inference::serve_embed_table`, the
//!   real `generate_text_embeddings` call the `model_inference`/
//!   `gpu_inference` tiers drive. Its span ends when the result table is
//!   COMMITTED — the Parquet object written, the ANN segment the embeddings
//!   sink builds beside it (a `usearch` HNSW graph, filled row by row)
//!   built, the catalog row flipped to ready. A reference that stops at "the
//!   vectors are in a file" has done less; `reference/torch_encode.py`'s
//!   `--ann-index` builds the same graph so the two spans close on the same
//!   work.
//!
//! ## The corpus: seeded, variable-length
//!
//! Rows of one length tokenize to a dense batch and hide what padding costs.
//! [`build_corpus`] draws each row's word count from a fixed four-band mixture
//! (most rows a sentence or a paragraph, a tail several hundred words long),
//! keyed by `(seed, row index)` alone — integer arithmetic throughout, so the
//! corpus is byte-identical on every platform and an `n`-row corpus is a
//! prefix of every larger one. Keys are zero-padded, so the plan's key order
//! (`CAST(key AS Utf8)`) is the corpus's row order and a reference producer
//! that walks the file in order forwards the rows in the chunks the plan does.
//!
//! ## CPU-hermetic default, GPU-device-parameterized
//!
//! `gpu_device` flows straight into the session's `select_device` (`-1` /
//! `Device::Cpu` for the hermetic default this tier's own tests run under, a
//! real CUDA ordinal for the pod producer). With no `--model-dir` the tier
//! serves a compiled-in fixture: the shared `tiny_bert` bundle plus an
//! EXPLICIT `1_Pooling/config.json`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_ai::model::tokenizer::TokenizerWrapper;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};
use sha2::{Digest, Sha256};
use tokio::process::Command;

use crate::finetune_step::sha256_and_len;
use crate::model_inference::{
    fold_vectors, local_model_id, read_served_vectors, serve_embed_table, session_over,
    write_corpus, ModelInferenceSpec, Row, ServeShape, KEY_COLUMN,
};
use crate::report::{CorpusIdentity, EncodePoint, EncodeStepTier};
use crate::timing::{nearest_rank, per_second, CostFit, ServeStats};
use crate::vram::{nvidia_smi_memory_used, DeviceMemoryProbe, VramSampler};

/// The CI-hermetic default device: `Device::Cpu`. The pod producer overrides
/// [`EncodeStepParams::gpu_device`] with a real CUDA ordinal.
pub const CPU_HERMETIC_DEVICE: i32 = -1;

/// What the tier serves and how it measures.
#[derive(Debug, Clone)]
pub struct EncodeStepParams {
    /// The checkpoint directory to serve (`config.json`, `model.safetensors`,
    /// `tokenizer.json`, optionally `1_Pooling/config.json`); `None` serves
    /// the compiled-in fixture.
    pub model_dir: Option<PathBuf>,
    /// The sweep: the corpus row count of each point, in the order given.
    pub rows: Vec<usize>,
    /// The corpus generation seed.
    pub seed: u64,
    /// `[inference] batch_size` — rows per model forward.
    pub batch_size: usize,
    /// `[inference] partitions` — the plan's inference fan-out.
    pub partitions: usize,
    /// `[gpu] compute_precision` — the precision the model loads at unless its
    /// own `config.json` declares one. What it RESOLVED to is what the tier
    /// records, read off the loaded model.
    pub compute_precision: jammi_numerics::ComputePrecision,
    /// Warm serves before the measured ones at each point, discarded.
    pub warmup: usize,
    /// Measured serves at each point.
    pub iters: usize,
    /// The device ordinal the session resolves on: [`CPU_HERMETIC_DEVICE`]
    /// (`-1`) for the hermetic default, a real CUDA ordinal for the pod
    /// producer.
    pub gpu_device: i32,
    /// Where each point leaves what a reference producer reads: the corpus
    /// Parquet it served (`corpus_<rows>.parquet`), the vectors it persisted
    /// (`vectors_<rows>.parquet`), and — when it served the compiled-in
    /// fixture — that checkpoint (`model/`). `None` keeps them in a scratch
    /// directory that dies with the point.
    pub exchange_dir: Option<PathBuf>,
}

impl EncodeStepParams {
    fn serve_shape(&self) -> ServeShape {
        ServeShape {
            gpu_device: self.gpu_device,
            batch_size: self.batch_size,
            partitions: self.partitions,
            compute_precision: self.compute_precision,
        }
    }
}

/// The words a corpus row is drawn from: plain vocabulary any text tokenizer
/// splits into real tokens (random bytes would tokenize to mostly the unknown
/// token and make the forward trivial).
const WORDS: [&str; 48] = [
    "the",
    "of",
    "and",
    "in",
    "to",
    "is",
    "for",
    "with",
    "on",
    "that",
    "quantum",
    "computing",
    "superconducting",
    "systems",
    "topological",
    "error",
    "correction",
    "codes",
    "spin",
    "coherence",
    "trapped",
    "ion",
    "qubits",
    "lattice",
    "gauge",
    "theory",
    "simulator",
    "gene",
    "editing",
    "inherited",
    "disease",
    "ribosome",
    "structure",
    "protein",
    "synthesis",
    "mitochondrial",
    "cellular",
    "respiration",
    "enzyme",
    "kinetics",
    "metabolic",
    "pathways",
    "measurement",
    "model",
    "results",
    "between",
    "under",
    "across",
];

/// The word-count mixture a row's length is drawn from: `(share of rows in
/// parts per hundred, fewest words, most words)`. A title or a sentence, a
/// paragraph, a long passage, and a tail of documents long enough to reach a
/// small model's truncation bound.
const LENGTH_BANDS: [(u64, u64, u64); 4] = [(30, 3, 10), (40, 11, 30), (22, 31, 90), (8, 91, 300)];

/// SplitMix64: a bijective 64-bit mix, so distinct `(seed, row, draw)` inputs
/// give independent-looking draws with no shared generator state to advance
/// in order.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Row `index` of the seeded corpus — a function of `(seed, index)` alone.
fn corpus_row(seed: u64, index: usize) -> Row {
    let draw = |n: u64| splitmix64(splitmix64(seed) ^ splitmix64((index as u64) << 20 | n));
    let band_draw = draw(0) % 100;
    let (_, fewest, most) = LENGTH_BANDS
        .into_iter()
        .scan(0, |upto, (share, fewest, most)| {
            *upto += share;
            Some((*upto, fewest, most))
        })
        .find(|&(upto, _, _)| band_draw < upto)
        .unwrap_or(LENGTH_BANDS[LENGTH_BANDS.len() - 1]);
    let word_count = fewest + draw(1) % (most - fewest + 1);
    let text = (0..word_count)
        .map(|w| WORDS[(draw(2 + w) % WORDS.len() as u64) as usize])
        .collect::<Vec<_>>()
        .join(" ");
    Row {
        id: format!("row_{index:08}"),
        text,
    }
}

/// The seeded variable-length corpus — see this module's doc.
pub(crate) fn build_corpus(seed: u64, row_count: usize) -> Vec<Row> {
    (0..row_count).map(|i| corpus_row(seed, i)).collect()
}

/// The `1_Pooling/config.json` flags declaring MEAN pooling — the same
/// six-key shape `jammi-ai`'s own `pooling_config.rs` it-suite writes for
/// its `mean_pooling_config` fixture, and the mapping `candle.rs`'s
/// `pooling_from_config` documents (`pooling_mode_mean_tokens: true` ->
/// `Pooling::Mean`).
fn mean_pooling_flags() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": false,
        "pooling_mode_mean_tokens": true,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// The CLS-pooling twin of [`mean_pooling_flags`] — same six-key shape,
/// `pooling_mode_cls_token: true` instead. Test-only:
/// drives [`build_encode_model_dir_with_pooling`] to prove
/// `checkpoint_pooling_sha256`/`pooling` actually react to a fixture flip,
/// never a hand-typed expectation this crate never actually measures.
#[cfg(test)]
fn cls_pooling_flags() -> serde_json::Value {
    serde_json::json!({
        "pooling_mode_cls_token": true,
        "pooling_mode_mean_tokens": false,
        "pooling_mode_max_tokens": false,
        "pooling_mode_mean_sqrt_len_tokens": false,
        "pooling_mode_weightedmean_tokens": false,
        "pooling_mode_lasttoken": false,
    })
}

/// Build a fresh model dir at `dst`: copy the shared `tiny_bert` fixture's
/// three files (the SAME fixture `model_inference`'s embed lane serves),
/// then write an EXPLICIT `1_Pooling/config.json` carrying `pooling_flags`.
///
/// The explicit pooling config (never the bare `tiny_bert` fixture, which
/// ships with no `1_Pooling/` folder at all) is deliberate: a repo with no
/// pooling config resolves through `candle.rs`'s silent mean-pooling
/// fallback — a silent-identity ambiguity. Declaring the strategy
/// here means [`crate::report::EncodeStepTier::pooling`] records a value
/// this tier KNOWS the engine resolves to, not an inferred one. Parameterized
/// over the flags (`checkpoint_pooling_sha256_and_
/// pooling_move_together_when_the_fixture_flips_to_cls` drives this with the
/// test-only `cls_pooling_flags` to prove the two accessors this tier reads
/// actually react to the fixture, never a hand-typed expectation).
fn build_encode_model_dir_with_pooling(
    dst: &Path,
    pooling_flags: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    let fixture = ModelInferenceSpec::embed_model_dir();
    for name in ["config.json", "model.safetensors", "tokenizer.json"] {
        std::fs::copy(fixture.join(name), dst.join(name)).map_err(|e| -> Box<dyn std::error::Error> {
            format!("copying {name} from the shared tiny_bert fixture into the encode-step model dir: {e}").into()
        })?;
    }
    let pooling_dir = dst.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir)?;
    std::fs::write(
        pooling_dir.join("config.json"),
        serde_json::to_string(pooling_flags)?,
    )?;
    Ok(())
}

/// The compiled-in fixture a run with no `--model-dir` serves:
/// [`build_encode_model_dir_with_pooling`] with [`mean_pooling_flags`].
fn build_encode_model_dir(dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    build_encode_model_dir_with_pooling(dst, &mean_pooling_flags())
}

/// [`crate::report::EncodeStepTier::device_requested`]'s value — `"cpu"` for
/// a negative ordinal (the CI-hermetic default), `"cuda:<ordinal>"`
/// otherwise. A cheap, honest label derived straight from the same
/// `gpu_device` value threaded into `model_inference::session_over`
/// (`select_device`'s own convention: negative selects `Device::Cpu`), never
/// a second, independently-resolved reading. Computable BEFORE any compute
/// runs (this is exactly what makes it an identity field, unlike
/// [`resolved_device_name`] below).
fn requested_device_label(gpu_device: i32) -> String {
    if gpu_device < 0 {
        "cpu".to_string()
    } else {
        format!("cuda:{gpu_device}")
    }
}

/// [`crate::report::EncodeStepTier::device_name`]'s value — a POST-HOC
/// hardware fact, only knowable after the device resolved (so
/// PROVENANCE, never identity). `"cpu"` for the CI-hermetic
/// default; a real CUDA leg queries the actual device sub-class name off the
/// driver via `gpu_inference::cuda_device_name` — the SAME in-process
/// `cudarc` lookup that tier already performs, never a second,
/// independently-drifting hardware-name query.
///
/// Naming `gpu_device` here (the REQUESTED ordinal) rather than re-reading
/// the session's own resolved device is only honest because the silent-
/// CPU-fallback state this would otherwise transcribe is UNREPRESENTABLE by
/// the time this function runs: `run()` threads
/// `gpu_device` through `model_inference::session_over`, which sets
/// `gpu.require_gpu = gpu_device >= 0` on the session's `GpuConfig`; a `-cuda
/// N` leg whose ordinal the box cannot actually satisfy fails the FIRST
/// model load (`CandleBackend::load`'s `select_device(device_config)?`,
/// `backend/candle.rs`'s `gpu_unavailable` returning a typed
/// `JammiError::Gpu`) at `measure_point`'s model load — well before this
/// function is ever reached. So on every path that DOES reach here,
/// requested ordinal and actually-resolved device are the same value by
/// construction: there is no code path left in which `gpu_device` names a
/// CUDA ordinal the run did not truly execute on.
///
/// **Qualification**:
/// on a build compiled with `feature = "metal"` but not `"cuda"`,
/// `select_device` itself can genuinely SUCCEED for a `gpu_device >= 0`
/// request via its metal branch (real Metal hardware, mislabeled here as a
/// CUDA ordinal) — the "first model load fails" argument above does not
/// apply on that build. The guarantee still holds there, but for a
/// DIFFERENT reason: this function's `cuda_device_name` call unconditionally
/// errors on any `not(feature = "cuda")` build (see its stub in
/// `gpu_inference.rs`), aborting `run()` before a mismatched, Metal-resolved
/// device name could ever populate this field. Both mechanisms are needed to
/// state the full picture; neither alone covers every build.
fn resolved_device_name(gpu_device: i32) -> Result<String, Box<dyn std::error::Error>> {
    if gpu_device < 0 {
        Ok("cpu".to_string())
    } else {
        crate::gpu_inference::cuda_device_name(gpu_device as u32)
    }
}

/// [`crate::report::EncodeStepTier::checkpoint_pooling_sha256`]'s value:
/// `sha256_and_len` over `model_dir/1_Pooling/config.json`'s bytes when that
/// file exists, `None` when it doesn't — the SAME presence
/// gate `backend::candle::all_candidate_paths` applies before hashing this
/// file into the engine's own `content_digest`
/// (`resolved.pooling_config.is_some()`), never a second, independently-
/// drifting presence check. Extracted to its own function so
/// `checkpoint_pooling_sha256_is_none_when_absent`/`..._is_some_when_present`
/// can drive the SAME code `run()` calls, rather than a re-typed copy that
/// could silently drift from it.
fn checkpoint_pooling_sha256(
    model_dir: &Path,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let pooling_config_path = model_dir.join("1_Pooling").join("config.json");
    if pooling_config_path.exists() {
        Ok(Some(sha256_and_len(&pooling_config_path)?.0))
    } else {
        Ok(None)
    }
}

/// What a real tokenization of one point's corpus measured: each row's real
/// (unpadded) token count in row order, and the tokens the plan's chunks pad
/// them out to.
struct TokenCounts {
    row_tokens: Vec<usize>,
    padded_tokens: usize,
}

/// Tokenize `rows` the way the plan forwards them: `batch_size` chunks in row
/// order, each truncated at `max_sequence_length` and padded to its own
/// longest row.
fn count_tokens(
    tokenizer: &TokenizerWrapper,
    rows: &[Row],
    batch_size: usize,
    max_sequence_length: usize,
) -> Result<TokenCounts, Box<dyn std::error::Error>> {
    rows.chunks(batch_size).try_fold(
        TokenCounts {
            row_tokens: Vec::with_capacity(rows.len()),
            padded_tokens: 0,
        },
        |mut counts, chunk| {
            let texts: Vec<&str> = chunk.iter().map(|r| r.text.as_str()).collect();
            let encoding = tokenizer.encode_batch(&texts, Some(max_sequence_length))?;
            counts.padded_tokens += encoding.seq_len * chunk.len();
            counts.row_tokens.extend(
                encoding
                    .attention_masks
                    .iter()
                    .map(|mask| mask.iter().map(|&bit| bit as usize).sum::<usize>()),
            );
            Ok(counts)
        },
    )
}

/// sha256 (hex) of the per-row token counts, in row order, as their decimal
/// renderings joined by `,` — a rendering any producer in any language
/// reproduces byte for byte. Two producers that tokenized and truncated the
/// same corpus the same way agree on it; the full vector (one entry per row
/// of a sweep point) never has to travel.
fn token_lengths_sha256(row_tokens: &[usize]) -> String {
    let rendered = row_tokens
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    hex::encode(Sha256::digest(rendered.as_bytes()))
}

/// Write the vectors a point persisted, keyed, for a reference producer to
/// compare against: `_row_id` beside a `vector` `FixedSizeList<Float32>`.
///
/// `vectors` is the served table's storage order, which for the never-refreshed
/// table a point serves is the plan's key order — and this corpus's keys sort
/// in row order — so row `i` of `rows` owns vector `i`.
async fn write_vectors(
    rows: &[Row],
    vectors: &[Vec<f32>],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let dim = vectors.first().map_or(0, Vec::len);
    if vectors.len() != rows.len() || vectors.iter().any(|v| v.len() != dim) {
        return Err(format!(
            "the served table holds {} vectors for {} corpus rows, or vectors of mixed width",
            vectors.len(),
            rows.len()
        )
        .into());
    }
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::clone(&item), dim as i32),
            false,
        ),
    ]));
    let values = Float32Array::from_iter_values(vectors.iter().flatten().copied());
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.id.as_str()),
            )) as ArrayRef,
            Arc::new(FixedSizeListArray::try_new(
                item,
                dim as i32,
                Arc::new(values),
                None,
            )?),
        ],
    )?;
    let url = StorageUrl::parse(path.to_str().ok_or("vectors path is not valid UTF-8")?)?;
    let handle = StorageRegistry::new().handle_for(&url, None)?;
    let mut writer = ObjectParquetWriter::open(&handle, schema).await?;
    writer.write_batch(&batch).await?;
    writer.close().await?;
    Ok(())
}

/// The device-memory probe a leg on `gpu_device` samples: the whole-device
/// `nvidia-smi` reading on a CUDA leg, nothing on a CPU leg (whose model never
/// touches a device).
fn device_memory_probe(gpu_device: i32) -> DeviceMemoryProbe {
    if gpu_device >= 0 {
        nvidia_smi_memory_used
    } else {
        || None
    }
}

/// Run the encode-step tier: one point measured in this process, or each point
/// of a sweep measured in a child of this binary and folded — see the module
/// doc's "One process per sweep point".
///
/// On a `--cuda N` leg (`params.gpu_device >= 0`) whose ordinal the box
/// cannot actually satisfy, this returns `Err` — never an `Ok(EncodeStepTier)`
/// carrying a `device_name` for hardware the run never touched: the session
/// sets `gpu.require_gpu = true` for that leg, so the model load fails with a
/// typed `JammiError::Gpu` instead of `select_device` silently degrading to
/// `Device::Cpu`. See [`resolved_device_name`]'s own doc for why this makes
/// `device_name` trustworthy on every path that DOES reach the end of a run.
pub async fn run(params: EncodeStepParams) -> Result<EncodeStepTier, Box<dyn std::error::Error>> {
    if params.rows.contains(&0) || params.batch_size == 0 || params.iters == 0 {
        return Err(format!(
            "encode-step needs every row count, the batch size and the measured iterations to be              at least 1: rows {:?}, batch size {}, iters {}",
            params.rows, params.batch_size, params.iters
        )
        .into());
    }
    let tier = match params.rows.as_slice() {
        [] => return Err("encode-step needs at least one row count".into()),
        [rows] => measure_point(&params, *rows).await?,
        sweep => {
            let mut children = Vec::with_capacity(sweep.len());
            for &rows in sweep {
                children.push(spawn_point(&params, rows).await?);
            }
            fold_sweep(children)?
        }
    };

    // Identity completeness, enforced on every real run (mirrors
    // `finetune_step::run`/`grad_oracle::run`'s own posture) — see
    // `report::assert_identity_fields_present`'s own doc.
    let value = serde_json::to_value(&tier)?;
    crate::report::assert_identity_fields_present(&value, EncodeStepTier::IDENTITY_FIELDS);
    crate::report::assert_identity_fields_present(&value, EncodeStepTier::PROVENANCE_FIELDS);
    Ok(tier)
}

/// Fold the one-point tiers of a sweep's children into the sweep's tier: their
/// points and corpus identities in sweep order, the two fits over them, and
/// everything else — which every child must have agreed on — from the first.
fn fold_sweep(children: Vec<EncodeStepTier>) -> Result<EncodeStepTier, Box<dyn std::error::Error>> {
    let shared = |tier: &EncodeStepTier| -> Result<serde_json::Value, serde_json::Error> {
        let mut value = serde_json::to_value(tier)?;
        if let Some(fields) = value.as_object_mut() {
            fields.retain(|name, _| !EncodeStepTier::PER_POINT_FIELDS.contains(&name.as_str()));
        }
        Ok(value)
    };
    let mut children = children.into_iter();
    let mut tier = children.next().ok_or("an empty sweep has no tier")?;
    let agreed = shared(&tier)?;
    for child in children {
        let measured = shared(&child)?;
        if measured != agreed {
            return Err(format!(
                "sweep points disagree on what they measured: {measured} vs {agreed}"
            )
            .into());
        }
        tier.rows.extend(child.rows);
        tier.corpus.extend(child.corpus);
        tier.points.extend(child.points);
    }
    let fit = |serve_ms: fn(&EncodePoint) -> f64| {
        let points: Vec<(usize, f64)> = tier.points.iter().map(|p| (p.rows, serve_ms(p))).collect();
        CostFit::least_squares(&points)
    };
    let (fit_p50, fit_min) = (fit(|p| p.serve_ms_p50), fit(|p| p.serve_ms_min));
    tier.fit_p50 = fit_p50;
    tier.fit_min = fit_min;
    Ok(tier)
}

/// Measure one sweep point in a child of this binary — the same subcommand
/// under the same flags, given the one row count — and read the tier off the
/// report it prints. The child's stderr is inherited so a failure surfaces in
/// the parent's log.
async fn spawn_point(
    params: &EncodeStepParams,
    rows: usize,
) -> Result<EncodeStepTier, Box<dyn std::error::Error>> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("encode-step")
        .args(["--rows", &rows.to_string()])
        .args(["--seed", &params.seed.to_string()])
        .args(["--batch-size", &params.batch_size.to_string()])
        .args(["--partitions", &params.partitions.to_string()])
        .args(["--compute-precision", &params.compute_precision.to_string()])
        .args(["--warmup", &params.warmup.to_string()])
        .args(["--iters", &params.iters.to_string()]);
    if params.gpu_device >= 0 {
        command.args(["--cuda", &params.gpu_device.to_string()]);
    }
    if let Some(model_dir) = &params.model_dir {
        command.arg("--model-dir").arg(model_dir);
    }
    if let Some(exchange_dir) = &params.exchange_dir {
        command.arg("--exchange-dir").arg(exchange_dir);
    }
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await?;
    if !output.status.success() {
        return Err(format!(
            "encode-step child ({rows} rows) exited with {}",
            output.status
        )
        .into());
    }
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    Ok(serde_json::from_value(
        report["tiers"]["encode_step"].clone(),
    )?)
}

/// Measure one row count in this process: write the corpus, stand the session
/// up over it, load the model, serve cold, serve warm, and assemble a
/// one-point tier.
async fn measure_point(
    params: &EncodeStepParams,
    row_count: usize,
) -> Result<EncodeStepTier, Box<dyn std::error::Error>> {
    let rows = build_corpus(params.seed, row_count);

    let scratch = tempfile::tempdir()?;
    let exchange_dir = params.exchange_dir.as_deref().unwrap_or(scratch.path());
    std::fs::create_dir_all(exchange_dir)?;
    let model_dir = match &params.model_dir {
        Some(dir) => dir.clone(),
        None => {
            let dir = exchange_dir.join("model");
            build_encode_model_dir(&dir)?;
            dir
        }
    };
    let model_id = local_model_id(&model_dir)?;

    let (checkpoint_config_sha256, _config_len) = sha256_and_len(&model_dir.join("config.json"))?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&model_dir.join("model.safetensors"))?;
    let (checkpoint_tokenizer_sha256, _tokenizer_len) =
        sha256_and_len(&model_dir.join("tokenizer.json"))?;
    let checkpoint_pooling_sha256 = checkpoint_pooling_sha256(&model_dir)?;

    let corpus_path = exchange_dir.join(format!("corpus_{row_count}.parquet"));
    write_corpus(&rows, &corpus_path).await?;
    let (corpus_sha256, _corpus_len) = sha256_and_len(&corpus_path)?;

    let session = session_over(&corpus_path, scratch.path(), params.serve_shape()).await?;

    // The model is loaded apart from the first serve so the device-memory
    // baseline can be read with the weights resident and nothing else: the
    // sampled growth is then activations and workspace, the quantity a
    // reference producer's allocator delta also measures.
    let load_start = std::time::Instant::now();
    let model_guard = session
        .model_cache()
        .get_or_load(
            &ModelSource::parse(&model_id),
            ModelTask::TextEmbedding,
            None,
        )
        .await?;
    let model_load_ms = load_start.elapsed().as_secs_f64() * 1_000.0;
    let compute_precision = model_guard.model.compute_precision().to_string();
    let pooling = model_guard
        .model
        .resolved_pooling()
        .map(|p| p.to_string())
        .unwrap_or_else(|| "none".to_string());
    let max_sequence_length = model_guard
        .model
        .max_sequence_length()
        .ok_or("the loaded model has no text forward to encode with")?;
    drop(model_guard);

    let probe = device_memory_probe(params.gpu_device);
    let vram_baseline = probe().unwrap_or(0);
    let sampler = VramSampler::start(probe);

    let (_table, first_serve_ms) = serve_embed_table(&session, &model_id).await?;
    for _ in 0..params.warmup {
        serve_embed_table(&session, &model_id).await?;
    }
    let mut serve_ms = Vec::with_capacity(params.iters);
    let mut served = None;
    for _ in 0..params.iters {
        let (table, ms) = serve_embed_table(&session, &model_id).await?;
        serve_ms.push(ms);
        served = Some(table);
    }
    let peak_vram_delta_bytes = sampler.and_then(|s| s.finish(vram_baseline).value);
    let stats = ServeStats::of(&serve_ms).ok_or("encode-step needs at least one measured serve")?;

    let vectors = read_served_vectors(
        &session,
        served.ok_or("encode-step needs at least one measured serve")?,
    )
    .await?;
    if params.exchange_dir.is_some() {
        write_vectors(
            &rows,
            &vectors,
            &exchange_dir.join(format!("vectors_{row_count}.parquet")),
        )
        .await?;
    }

    // Real tokenization off the model's own `tokenizer.json`, through the SAME
    // wrapper the candle backend loads — see this module's own doc.
    let tokenizer = TokenizerWrapper::from_file(&model_dir.join("tokenizer.json"))?;
    let counts = count_tokens(&tokenizer, &rows, params.batch_size, max_sequence_length)?;
    let tokens: usize = counts.row_tokens.iter().sum();
    let mut sorted_tokens: Vec<f64> = counts.row_tokens.iter().map(|&n| n as f64).collect();
    sorted_tokens.sort_by(|a, b| a.total_cmp(b));

    Ok(EncodeStepTier {
        seed: params.seed,
        rows: vec![row_count],
        batch_size: params.batch_size,
        corpus: vec![CorpusIdentity {
            rows: row_count,
            corpus_sha256,
            token_lengths_sha256: token_lengths_sha256(&counts.row_tokens),
            tokens,
        }],
        max_sequence_length,
        compute_precision,
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        checkpoint_tokenizer_sha256,
        pooling,
        // `jammi_encoders::pool_and_normalize` mandatorily L2-normalizes on
        // every reachable path — see `EncodeStepTier::normalize`'s own doc.
        normalize: true,
        warmup: params.warmup,
        iters_measured: params.iters,
        checkpoint_pooling_sha256,
        device_requested: requested_device_label(params.gpu_device),
        partitions: params.partitions,
        model_dir: params
            .model_dir
            .as_ref()
            .map(|dir| dir.display().to_string()),
        device_name: resolved_device_name(params.gpu_device)?,
        kernels_disabled_requested: jammi_kernels::admission::disabled_ops_requested(),
        kernels_disabled_fired: jammi_kernels::admission::disabled_ops_fired(),
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        build_features: crate::report::build_features()
            .into_iter()
            .map(str::to_string)
            .collect(),
        // The encode/eval path has no chunked-attention arm at all — see
        // `EncodeStepTier::chunk_size`'s own doc.
        chunk_size: None,
        // Fused attention arms are training-only; the encode/eval path
        // always runs eager. See `EncodeStepTier`'s own doc for why this is
        // provenance, never identity.
        attention_arm: "eager".to_string(),
        points: vec![EncodePoint {
            rows: row_count,
            padded_tokens: counts.padded_tokens,
            row_tokens_p50: nearest_rank(&sorted_tokens, 0.50) as usize,
            row_tokens_max: sorted_tokens.last().map_or(0, |&n| n as usize),
            vectors_digest: fold_vectors(&vectors),
            model_load_ms,
            first_serve_ms,
            serve_ms_p50: stats.p50_ms,
            serve_ms_min: stats.min_ms,
            rows_per_s: per_second(row_count, stats.p50_ms),
            tokens_per_s: per_second(tokens, stats.p50_ms),
            peak_rss_bytes: crate::rss::peak_rss_bytes(),
            peak_vram_delta_bytes,
        }],
        fit_p50: None,
        fit_min: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_inference::corpus_session;

    fn test_params() -> EncodeStepParams {
        EncodeStepParams {
            model_dir: None,
            rows: vec![48],
            seed: 0,
            batch_size: 8,
            partitions: 1,
            compute_precision: jammi_numerics::ComputePrecision::F32,
            warmup: 1,
            iters: 2,
            gpu_device: CPU_HERMETIC_DEVICE,
            exchange_dir: None,
        }
    }

    /// Cardinality pin: the EXACT comparison identity set, in this exact
    /// order — so `ci/scripts/perf/identity_fields.py`'s
    /// `ENCODE_IDENTITY_FIELDS` has a fixed, reviewable Rust-side source to
    /// mirror. A field added, removed, or renamed here is a visible, reviewed
    /// diff against this test, not a silent drift the Python mirror would only
    /// notice indirectly.
    #[test]
    fn identity_fields_cardinality_is_pinned() {
        let names: Vec<&str> = EncodeStepTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "seed",
                "rows",
                "batch_size",
                "corpus",
                "max_sequence_length",
                "compute_precision",
                "checkpoint_config_sha256",
                "checkpoint_weights_sha256",
                "checkpoint_weights_size_bytes",
                "checkpoint_tokenizer_sha256",
                "pooling",
                "normalize",
                "warmup",
                "iters_measured",
                "checkpoint_pooling_sha256",
                "device_requested",
            ],
            "EncodeStepTier::IDENTITY_FIELDS drifted — update this pin together with              ci/scripts/perf/identity_fields.py's ENCODE_IDENTITY_FIELDS"
        );
    }

    /// The forbidden-in-identity clause as a checked negative control, not
    /// only prose: `attention_arm` — and every other declared provenance
    /// field — must never appear in `IDENTITY_FIELDS`, mechanically enforced
    /// so a future "helpful" addition trips a test instead of silently
    /// introducing a false determinant.
    #[test]
    fn provenance_fields_are_never_members_of_identity_fields() {
        let identity_names: std::collections::HashSet<&str> = EncodeStepTier::IDENTITY_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        for (provenance_name, _) in EncodeStepTier::PROVENANCE_FIELDS {
            assert!(
                !identity_names.contains(provenance_name),
                "{provenance_name:?} is a declared PROVENANCE_FIELDS entry but also \
                 appears in IDENTITY_FIELDS — attention_arm/chunk_size/device_name/\
                 kernels_disabled_*/flash_compiled/build_features are forbidden from \
                 identity on this surface"
            );
        }
        assert!(
            EncodeStepTier::PROVENANCE_FIELDS
                .iter()
                .any(|(name, _)| *name == "attention_arm"),
            "attention_arm must be declared as a PROVENANCE_FIELDS entry"
        );
    }

    /// The provenance roster's own name pin.
    #[test]
    fn provenance_fields_cardinality_is_pinned() {
        let names: Vec<&str> = EncodeStepTier::PROVENANCE_FIELDS
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "partitions",
                "model_dir",
                "device_name",
                "kernels_disabled_requested",
                "kernels_disabled_fired",
                "flash_compiled",
                "build_features",
                "chunk_size",
                "attention_arm",
            ]
        );
    }

    /// The corpus is a function of `(seed, row index)` alone: an `n`-row
    /// corpus is a prefix of a larger one, a different seed is a different
    /// corpus, and the zero-padded keys sort in row order — the property that
    /// makes the plan's key order the corpus's own.
    #[test]
    fn the_corpus_is_seeded_prefix_stable_and_keyed_in_row_order() {
        let small = build_corpus(7, 64);
        let large = build_corpus(7, 300);
        assert!(small
            .iter()
            .zip(&large)
            .all(|(a, b)| a.id == b.id && a.text == b.text));
        assert!(build_corpus(8, 64)
            .iter()
            .zip(&small)
            .any(|(a, b)| a.text != b.text));
        let mut sorted: Vec<&str> = large.iter().map(|r| r.id.as_str()).collect();
        sorted.sort_unstable();
        assert!(sorted.iter().zip(&large).all(|(id, row)| *id == row.id));
    }

    /// The length mixture covers every row (its shares are parts per hundred)
    /// and really is variable: a few hundred rows span an order of magnitude
    /// in word count, with every band's range respected.
    #[test]
    fn the_corpus_length_mixture_is_whole_and_variable() {
        assert_eq!(LENGTH_BANDS.iter().map(|b| b.0).sum::<u64>(), 100);
        let words: Vec<usize> = build_corpus(0, 400)
            .iter()
            .map(|r| r.text.split(' ').count())
            .collect();
        let (fewest, most) = (
            *words.iter().min().expect("rows"),
            *words.iter().max().expect("rows"),
        );
        assert!(fewest >= 3 && most <= 300, "{fewest}..{most}");
        assert!(
            most >= 10 * fewest,
            "400 rows must reach both ends of the mixture: {fewest}..{most}"
        );
    }

    /// The token-length digest is the sha256 of the decimal counts joined by
    /// commas — pinned against a value computed outside this crate, the same
    /// known answer `reference/torch_encode.py`'s own test pins, so the two
    /// producers cannot drift apart on the rendering.
    #[test]
    fn token_lengths_sha256_is_the_pinned_rendering() {
        assert_eq!(
            token_lengths_sha256(&[3, 5, 8]),
            "d8c85d93367f9ebb51148c0a399b58ca7386e2de1a4ffcc5e925048f6d4e2250"
        );
    }

    /// The teeth, GATE-FAILS direction (an assertion must be able to fail):
    /// `run()` drives the REAL serving surface end to end on `Device::Cpu` —
    /// real tokenization, real checksums, real `generate_text_embeddings`
    /// serves — and every declared identity AND provenance field lands
    /// populated on the emitted tier.
    #[tokio::test(flavor = "multi_thread")]
    async fn encode_step_drives_the_real_surface_and_populates_every_field() {
        let params = test_params();
        let tier = run(params.clone()).await.expect("encode-step run");

        assert_eq!(tier.seed, params.seed);
        assert_eq!(tier.rows, params.rows);
        assert_eq!(tier.batch_size, params.batch_size);
        assert_eq!(tier.partitions, params.partitions);
        assert_eq!(tier.warmup, params.warmup);
        assert_eq!(tier.iters_measured, params.iters);
        assert_eq!(tier.model_dir, None, "the compiled-in fixture has no path");
        assert_eq!(tier.max_sequence_length, 128, "tiny_bert's position bound");
        assert_eq!(tier.compute_precision, "f32");
        assert_eq!(tier.pooling, "mean");
        assert!(tier.normalize);
        assert_eq!(tier.attention_arm, "eager");
        assert_eq!(tier.chunk_size, None);
        assert_eq!(tier.device_name, "cpu");
        assert_eq!(tier.device_requested, "cpu");
        for (digest, name) in [
            (&tier.checkpoint_config_sha256, "config"),
            (&tier.checkpoint_weights_sha256, "weights"),
            (&tier.checkpoint_tokenizer_sha256, "tokenizer"),
            (
                tier.checkpoint_pooling_sha256
                    .as_ref()
                    .expect("this tier's fixture always carries 1_Pooling/config.json"),
                "pooling config",
            ),
            (&tier.corpus[0].corpus_sha256, "corpus"),
            (&tier.corpus[0].token_lengths_sha256, "token lengths"),
        ] {
            assert_eq!(digest.len(), 64, "{name} sha256 must be 64 hex chars");
            assert!(
                digest.chars().all(|c| c.is_ascii_hexdigit()),
                "{name} sha256 must be hex"
            );
        }
        assert!(tier.checkpoint_weights_size_bytes > 0);

        let [point] = tier.points.as_slice() else {
            panic!("one row count is one point: {:?}", tier.points);
        };
        assert_eq!(point.rows, 48);
        assert_eq!(tier.corpus[0].rows, 48);
        // The teeth: a corpus of genuinely different lengths must pad — a real
        // tokenization, not a dense assumption — and its long tail must reach
        // the model's truncation bound without passing it.
        assert!(
            point.padded_tokens > tier.corpus[0].tokens,
            "variable-length rows must cost padding: {} padded vs {} real",
            point.padded_tokens,
            tier.corpus[0].tokens
        );
        assert!(point.row_tokens_p50 < point.row_tokens_max);
        assert!(point.row_tokens_max <= tier.max_sequence_length);
        assert!(point.serve_ms_min > 0.0 && point.serve_ms_min <= point.serve_ms_p50);
        assert!(point.first_serve_ms > 0.0 && point.model_load_ms > 0.0);
        assert!(point.rows_per_s > 0.0 && point.tokens_per_s > point.rows_per_s);
        assert_eq!(
            point.peak_vram_delta_bytes, None,
            "a CPU leg samples no device"
        );
        assert_eq!(point.vectors_digest.len(), 16);
        assert_eq!(
            (tier.fit_p50, tier.fit_min),
            (None, None),
            "one point fits nothing"
        );
    }

    /// `[inference] partitions` is a fan-out, never an input: the vectors a
    /// point persists are byte-identical at `partitions = 1` and
    /// `partitions = 4`, over a corpus wide enough (more chunks than
    /// partitions) for the fan-out to really split it. This is what lets
    /// `partitions` sit in provenance — two legs that differ only in it
    /// measured the same outputs.
    #[tokio::test(flavor = "multi_thread")]
    async fn partitions_never_changes_the_persisted_vectors() {
        let serial = run(test_params()).await.expect("partitions = 1");
        let fanned = run(EncodeStepParams {
            partitions: 4,
            ..test_params()
        })
        .await
        .expect("partitions = 4");
        assert_eq!(fanned.partitions, 4);
        assert_eq!(serial.corpus, fanned.corpus);
        assert_eq!(
            serial.points[0].vectors_digest,
            fanned.points[0].vectors_digest
        );
    }

    /// A sweep folds its children's one-point tiers into one: points and
    /// corpus identities in sweep order, both fits present, the shared
    /// identity carried once — and a child that measured under a different
    /// premise is refused, never folded.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_sweep_folds_its_points_and_refuses_a_point_under_another_premise() {
        let point = |rows: usize, batch_size: usize| {
            run(EncodeStepParams {
                rows: vec![rows],
                batch_size,
                ..test_params()
            })
        };
        let (small, large) = (
            point(16, 8).await.expect("16 rows"),
            point(96, 8).await.expect("96 rows"),
        );
        let tier = fold_sweep(vec![small, large]).expect("fold");
        assert_eq!(tier.rows, vec![16, 96]);
        assert_eq!(
            tier.points.iter().map(|p| p.rows).collect::<Vec<_>>(),
            vec![16, 96]
        );
        assert_eq!(
            tier.corpus.iter().map(|c| c.rows).collect::<Vec<_>>(),
            vec![16, 96]
        );
        let fit = tier.fit_min.expect("two row counts determine two terms");
        assert!(
            fit.relative_residual_rms < 1e-9,
            "two points fit exactly: {fit:?}"
        );
        assert!(tier.fit_p50.is_some());

        let other_premise = point(96, 4).await.expect("96 rows at another batch size");
        let refused = fold_sweep(vec![point(16, 8).await.expect("16 rows"), other_premise])
            .expect_err("a point served at another batch size is another measurement");
        assert!(refused.to_string().contains("disagree"), "{refused}");
    }

    /// The exchange directory carries what a reference producer reads: the
    /// corpus the point served, byte for byte the file its `corpus_sha256`
    /// names, the vectors it persisted, and the fixture checkpoint it served.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_point_leaves_its_corpus_and_vectors_in_the_exchange_directory() {
        let exchange = tempfile::tempdir().expect("tempdir");
        let tier = run(EncodeStepParams {
            exchange_dir: Some(exchange.path().to_path_buf()),
            ..test_params()
        })
        .await
        .expect("encode-step run");
        let (corpus_sha256, _) =
            sha256_and_len(&exchange.path().join("corpus_48.parquet")).expect("corpus file");
        assert_eq!(corpus_sha256, tier.corpus[0].corpus_sha256);
        assert!(exchange.path().join("vectors_48.parquet").exists());
        let (weights_sha256, _) =
            sha256_and_len(&exchange.path().join("model").join("model.safetensors"))
                .expect("the fixture checkpoint the point served");
        assert_eq!(weights_sha256, tier.checkpoint_weights_sha256);
    }

    /// A run that could measure nothing is refused before it builds anything.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_with_nothing_to_measure_is_refused() {
        for params in [
            EncodeStepParams {
                rows: vec![],
                ..test_params()
            },
            EncodeStepParams {
                rows: vec![16, 0],
                ..test_params()
            },
            EncodeStepParams {
                iters: 0,
                ..test_params()
            },
        ] {
            run(params).await.expect_err("nothing to measure");
        }
    }

    /// The teeth for `checkpoint_tokenizer_sha256`: a run
    /// against a model dir whose `tokenizer.json` bytes differ from the
    /// fixture's own, with `config.json`/`model.safetensors` held byte-
    /// identical, must move the recorded tokenizer digest (and ONLY that
    /// digest) — proving the field is a real content hash of the actual
    /// tokenizer bytes served, not a copy of `checkpoint_config_sha256` or a
    /// constant. Perturbs the SAME fixture `build_encode_model_dir`
    /// produces (rather than driving a second `run()`, which would also be
    /// legitimate but slower) so the assertion isolates the one changed
    /// file.
    #[test]
    fn checkpoint_tokenizer_sha256_reacts_to_the_actual_tokenizer_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let (tokenizer_baseline, _) =
            sha256_and_len(&dir.path().join("tokenizer.json")).expect("hash tokenizer.json");
        let (config_baseline, _) =
            sha256_and_len(&dir.path().join("config.json")).expect("hash config.json");

        // Perturb: append a byte to tokenizer.json only; config.json/
        // model.safetensors are never touched.
        let mut bytes = std::fs::read(dir.path().join("tokenizer.json")).expect("read tokenizer");
        bytes.push(b'\n');
        std::fs::write(dir.path().join("tokenizer.json"), &bytes).expect("write perturbed");
        let (tokenizer_perturbed, _) =
            sha256_and_len(&dir.path().join("tokenizer.json")).expect("hash perturbed tokenizer");
        let (config_after, _) =
            sha256_and_len(&dir.path().join("config.json")).expect("hash config.json again");

        assert_ne!(
            tokenizer_baseline, tokenizer_perturbed,
            "a byte-perturbed tokenizer.json must move its own sha256"
        );
        assert_eq!(
            config_baseline, config_after,
            "an untouched config.json must keep the same sha256 — the perturbation isolated \
             to tokenizer.json alone"
        );
    }

    /// The explicit `1_Pooling/config.json` fixture this tier builds
    /// resolves to a DIFFERENT pooled vector than the bare `tiny_bert`
    /// fixture (no `1_Pooling/` folder, candle's silent mean fallback would
    /// otherwise make this indistinguishable) — proving the declared
    /// config is actually consumed, not merely present alongside an
    /// unrelated default. This is the SAME shape `jammi-ai`'s own
    /// `pooling_config.rs` proves at the engine layer;
    /// here it only proves the FIXTURE is wired to a real, distinguishing
    /// config — the identity-hash-folds-it proof lives in `jammi-ai`'s own
    /// suite (out of this crate's scope).
    #[test]
    fn encode_model_dir_carries_an_explicit_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let pooling_json = dir.path().join("1_Pooling").join("config.json");
        assert!(
            pooling_json.exists(),
            "1_Pooling/config.json must be written"
        );
        let raw = std::fs::read_to_string(&pooling_json).expect("read pooling config");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(value["pooling_mode_mean_tokens"], serde_json::json!(true));
        assert_eq!(value["pooling_mode_cls_token"], serde_json::json!(false));
    }

    /// A `--cuda N` leg (`gpu_device >= 0`)
    /// on a box with no usable CUDA device must REFUSE with a typed error
    /// rather than silently degrading to `Device::Cpu` and publishing
    /// `device_requested:"cuda:0"` plus a real `device_name` for a run that
    /// actually executed on CPU (`resolved_device_name` queries
    /// `cuda_device_name` off the REQUESTED ordinal alone, so the session's
    /// `require_gpu` is what enforces that it actually resolved CUDA). This
    /// test runs on the CPU-hermetic default
    /// build (`not(feature = "cuda")`) — a box with no CUDA backend compiled
    /// in is exactly the "GPU-less box" this refusal must hold on, so it
    /// exercises the real `require_gpu` failure mode with no mocking.
    /// `#[cfg(not(feature = "cuda"))]` mirrors `GpuScheduler`'s own
    /// `for_device_falls_back_to_unlimited_when_probe_fails`: a `cuda`-feature
    /// build with a real device at ordinal 0 would legitimately succeed here,
    /// which is a different (valid) case this test is not about.
    #[cfg(not(feature = "cuda"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn cuda_leg_on_a_gpu_less_box_refuses_and_emits_no_report() {
        let params = EncodeStepParams {
            gpu_device: 0,
            ..test_params()
        };
        let err = run(params).await.expect_err(
            "a --cuda 0 leg on a box with no usable CUDA device must refuse (typed error), \
             never silently serve on CPU and emit a report",
        );
        // The typed refusal threaded via `model_inference::session_over`'s
        // `require_gpu = gpu_device >= 0` (`gpu_unavailable`'s
        // `JammiError::Gpu`, `#[error("GPU error: {0}")]`, message contains
        // "GPU required") — not some unrelated failure (a missing fixture, a
        // tokenizer error) that would also make this test spuriously pass.
        let message = err.to_string();
        assert!(
            message.contains("GPU required"),
            "expected the typed require_gpu refusal (JammiError::Gpu), got: {message}"
        );
    }

    #[test]
    fn requested_device_label_reads_cpu_for_negative_ordinal_and_cuda_otherwise() {
        assert_eq!(requested_device_label(-1), "cpu");
        assert_eq!(requested_device_label(0), "cuda:0");
        assert_eq!(requested_device_label(3), "cuda:3");
    }

    /// [`resolved_device_name`] on the CI-hermetic default never touches the
    /// `cuda`-feature-gated `cudarc` path at all (`device_name` is a
    /// POST-HOC hardware fact, but `"cpu"` is knowable
    /// without querying any driver).
    #[test]
    fn resolved_device_name_reads_cpu_for_negative_ordinal_without_a_cuda_query() {
        assert_eq!(resolved_device_name(-1).expect("cpu never errors"), "cpu");
    }

    /// `checkpoint_pooling_sha256` is `None` — never a
    /// panic, never an empty-string stand-in — for a model dir that carries
    /// no `1_Pooling/` folder at all, the SAME presence gate
    /// `backend::candle::all_candidate_paths` applies to the engine's own
    /// `content_digest`.
    #[test]
    fn checkpoint_pooling_sha256_is_none_when_the_model_dir_has_no_pooling_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fixture = ModelInferenceSpec::embed_model_dir();
        for name in ["config.json", "model.safetensors", "tokenizer.json"] {
            std::fs::copy(fixture.join(name), dir.path().join(name)).expect("copy fixture file");
        }
        assert!(
            !dir.path().join("1_Pooling").exists(),
            "the bare tiny_bert fixture ships with no 1_Pooling/ folder"
        );
        assert_eq!(
            checkpoint_pooling_sha256(dir.path()).expect("presence-gated read never errors"),
            None
        );
    }

    /// `checkpoint_pooling_sha256` is `Some(sha256_and_len(..).0)` when the
    /// file IS present — driving the SAME `sha256_and_len` helper directly
    /// against the fixture's own bytes, so this is a real cross-check
    /// against an independently-computed hash, never a self-referential
    /// assertion that the function returns whatever the function returns.
    #[test]
    fn checkpoint_pooling_sha256_is_some_and_matches_a_direct_hash_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        build_encode_model_dir(dir.path()).expect("build fixture model dir");
        let expected = sha256_and_len(&dir.path().join("1_Pooling").join("config.json"))
            .expect("hash pooling config directly")
            .0;
        assert_eq!(
            checkpoint_pooling_sha256(dir.path()).expect("presence-gated read never errors"),
            Some(expected)
        );
    }

    /// The pooling teeth test: flip the fixture's `1_Pooling/config.json` from
    /// MEAN to CLS, holding `config.json` byte-identical, and drive the REAL
    /// accessors this tier's `run()` reads (`LoadedModel::resolved_pooling`
    /// via the engine's own model cache, and `checkpoint_pooling_sha256`
    /// over the actual file bytes) — proving BOTH `pooling` and
    /// `checkpoint_pooling_sha256` move on the flip while
    /// `checkpoint_config_sha256` stays put. With `pooling` as a hardcoded
    /// `"mean"` literal and no field hashing the pooling-config bytes, this
    /// exact flip would leave every other identity field byte-identical
    /// while the served vectors differ.
    #[tokio::test(flavor = "multi_thread")]
    async fn checkpoint_pooling_sha256_and_pooling_move_together_when_the_fixture_flips_to_cls() {
        async fn resolved_pooling_and_hashes(
            pooling_flags: &serde_json::Value,
        ) -> (String, String, Option<String>) {
            let model_tmp = tempfile::tempdir().expect("tempdir");
            build_encode_model_dir_with_pooling(model_tmp.path(), pooling_flags)
                .expect("build fixture model dir");
            let model_id = local_model_id(model_tmp.path()).expect("model id");
            let (config_sha, _) =
                sha256_and_len(&model_tmp.path().join("config.json")).expect("hash config.json");
            let pooling_sha = checkpoint_pooling_sha256(model_tmp.path())
                .expect("presence-gated read never errors");

            let (session, _dir) = corpus_session(&build_corpus(0, 1), ServeShape::cpu())
                .await
                .expect("corpus session");
            let model_source = ModelSource::parse(&model_id);
            let model_guard = session
                .model_cache()
                .get_or_load(&model_source, ModelTask::TextEmbedding, None)
                .await
                .expect("load model");
            let pooling = model_guard
                .model
                .resolved_pooling()
                .map(|p| p.to_string())
                .unwrap_or_else(|| "none".to_string());
            (pooling, config_sha, pooling_sha)
        }

        let (mean_pooling, mean_config_sha, mean_pooling_sha) =
            resolved_pooling_and_hashes(&mean_pooling_flags()).await;
        let (cls_pooling, cls_config_sha, cls_pooling_sha) =
            resolved_pooling_and_hashes(&cls_pooling_flags()).await;

        assert_eq!(mean_pooling, "mean");
        assert_eq!(cls_pooling, "cls");
        assert_ne!(
            mean_pooling, cls_pooling,
            "pooling must move when the fixture flips mean->cls"
        );
        assert_ne!(
            mean_pooling_sha, cls_pooling_sha,
            "checkpoint_pooling_sha256 must move when the fixture flips mean->cls"
        );
        assert_eq!(
            mean_config_sha, cls_config_sha,
            "config.json is byte-identical across the flip -- only 1_Pooling/config.json changed"
        );
    }

    /// `jammi_encoders::pool_and_normalize` mandatorily L2-normalizes every
    /// reachable output — proved directly here by feeding it a hand-built,
    /// deliberately non-unit-norm hidden-state tensor (a large constant
    /// value at every position, no L2-normalization applied anywhere before
    /// this call) and asserting the pooled output still comes out unit-norm.
    /// A future signature change that added a normalize-optional toggle
    /// would either fail this call to compile (a new required parameter) or
    /// — if given a default that preserved today's always-normalize
    /// behavior — still leave this assertion green, so the pinned invariant
    /// this test protects is "there is no code path in this crate's
    /// dependency graph that reaches a pooled embedding without going
    /// through `pool_and_normalize`", the same claim `EncodeStepTier::normalize`'s
    /// own doc pins.
    #[test]
    fn pool_and_normalize_is_mandatory_with_no_toggle() {
        let hidden = candle_core::Tensor::full(7.0f32, (2, 3, 4), &candle_core::Device::Cpu)
            .expect("build a deliberately non-unit-norm hidden tensor");
        let attention_mask =
            candle_core::Tensor::ones((2, 3), candle_core::DType::U32, &candle_core::Device::Cpu)
                .expect("build a real-tokens-only attention mask");

        let pooled = jammi_encoders::pool_and_normalize(
            &hidden,
            &attention_mask,
            jammi_encoders::Pooling::Mean,
        )
        .expect("pool_and_normalize");

        let norms: Vec<f32> = pooled
            .sqr()
            .and_then(|t| t.sum(1))
            .and_then(|t| t.sqrt())
            .expect("compute per-row L2 norm")
            .to_vec1()
            .expect("read norms back to host");
        for norm in norms {
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "pool_and_normalize must always emit a unit-L2-norm row, got {norm}"
            );
        }
    }
}
