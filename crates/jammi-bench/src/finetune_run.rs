//! The finetune-run tier: one full fine-tune (seed, arm) run driving the REAL
//! `jammi_ai::fine_tune::trainer::TrainingLoopBuilder` and the public
//! per-example held-out seam (`TrainingLoop::evaluate_held_out`, unit 63 H1),
//! feeding CONTRACT H4/PLAN (d)'s finetune-run tier (unit 63, C16's phase-2
//! lift).
//!
//! ## Why this tier is a "heavier" build than [`crate::finetune_step`]
//!
//! [`crate::finetune_step`] measures ONE step's cost by driving the encoder +
//! optimizer directly, bypassing everything job-shaped: no catalog row, no
//! lease, no resume checkpoint. This tier drives the trainer's real PUBLIC
//! entry point end to end — `TrainingLoopBuilder::job_id`/`worker_id`/
//! `catalog`/`artifact_dir`/`artifact_store` are ALL required (`build()`
//! refuses without them, see that method's doc) — so a caller that wants the
//! real `run()`/`evaluate_held_out()` call graph must first stand up a real
//! (if CPU-hermetic, file-backed) `Catalog` + `ArtifactStore` and claim a real
//! training-job row, exactly as `fine_tune::worker::run_fine_tune_blocking`
//! does for a production job. That plumbing — not the training math — is
//! what the plan's closing budget note prices as "a heavier tier build".
//!
//! ## The full per-epoch trajectory: resume-cycling, not a callback
//!
//! `TrainingLoop::run()` has no per-epoch callback — CONTRACT H4 is explicit
//! that `TrainingLoopBuilder`'s surface is the "real construction surface"
//! this tier must drive, and this crate does not own `jammi-ai` (ai-core
//! domain), so adding a hook there is out of scope. The ONLY way to observe
//! `evaluate_held_out()` at an EPOCH BOUNDARY through the public surface
//! alone is the same mechanism a crash-and-resume across a process boundary
//! already uses: `run()` with `config.epochs = k+1` against a loop RESTORED
//! from the durable resume checkpoint `run()` itself writes at every epoch
//! boundary (unconditionally, when `artifact_store` is set —
//! `TrainingLoop::save_resume_checkpoint`'s call site in `run()`), executes
//! EXACTLY epoch `k`, and leaves the SAME `TrainingLoop` instance (still
//! `&mut`) with post-epoch-k weights this tier immediately calls
//! `evaluate_held_out` against. [`run`] below cycles through `params.epochs`
//! of these single-epoch, resume-chained legs — a real multi-process-style
//! resume repeated `epochs` times in one process, not a mock. Each new
//! `TrainingLoopBuilder` gets a FRESH `VarMap` + a freshly-constructed
//! LoRA-injected encoder (deterministic from `(seed, target_modules)`,
//! `LoraInitMode::ZerosB`); the resume restore (`TrainingLoop`'s internal
//! `restore_from_checkpoint`) then overwrites those fresh `Var`s BY NAME from
//! the persisted bundle — precisely the sequence a real crash-and-resume
//! exercises, so this tier's trajectory is not a smaller, easier substitute
//! measurement, it is the production continuity mechanism driven on a
//! schedule.
//!
//! [`FinetuneRunParams::eval_cadence`] controls how often `evaluate_held_out`
//! is called against the held-out fixture as this cycle advances (every
//! `eval_cadence` epochs, and unconditionally on the LAST epoch so the
//! FINAL-EPOCH endpoint is always present — CONTRACT H4/Frame: `d_i` = the
//! FINAL epoch's `evaluate_held_out().mean`, never
//! `TrainingResult::final_loss`, which is `best_val_loss`, a min-over-epochs
//! order statistic).
//!
//! ## Arm selection: provenance, not identity
//!
//! The fused-vs-ALLOFF arm is selected the SAME way every other kernel A/B
//! producer in this repo selects it: the `JAMMI_KERNELS_DISABLE` env var,
//! read once per process by `jammi_kernels::admission` and memoized in a
//! `OnceLock` (see [`crate::finetune_step`]'s own doc on
//! `attention_arm`/`kernels_disabled_requested`). This tier does not set
//! that env var itself — the CALLER sets `JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused`
//! (or leaves it unset for the fused arm) before invoking `jammi-bench
//! finetune-run`, mirroring `finetune-step`'s own convention exactly (a
//! fresh child PROCESS per leg is how the existing kernel-disable test suite
//! gets a fresh `OnceLock`). [`FinetuneRunTier::arm`] records what the
//! CALLER told this run to be (`--arm`), and
//! [`FinetuneRunTier::attention_arm`]/`kernels_disabled_requested` record
//! what the PROCESS actually resolved — both are PROVENANCE fields, never
//! identity: the C16/H2 sign test is a PAIRED comparison ACROSS arms (`d_i =
//! fused - alloff`, same seed), so a merger that treated the arm as identity
//! could never pair the two legs it exists to compare (CONTRACT H4: "the arm
//! is provenance, never identity").
//!
//! ## Held-out split is disjoint from the internal train/val split
//!
//! `TrainingDataLoader::split` (inside `TrainingLoop::run`) carves an early-
//! stopping validation slice OUT OF the rows this tier passes as its TRAIN
//! loader. The C16/H2 held-out fixture is a SEPARATE, disjoint loader that
//! never enters `run()` at all — it is fed ONLY to `evaluate_held_out`,
//! directly. This is the DISJOINT convention `EncodeStepTier` established
//! (E3's precedent, see that struct's own `PROVENANCE_FIELDS` doc) rather
//! than a superset: a row can be a member of the training set's internal val
//! split AND the C16 held-out set only by construction error, and disjointness
//! is enforced by the CALLER supplying two non-overlapping row sets (this
//! module does not itself check the two lists for overlap — the committed
//! fixture's own `train_ids_sha256.json` vs `heldout_ids.txt` partition is
//! the source of that guarantee, docs-ci/cookbook domain).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use candle_core::Device;
use candle_nn::VarMap;

use jammi_ai::fine_tune::data::TrainingDataLoader;
use jammi_ai::fine_tune::resume::load_bundle;
use jammi_ai::fine_tune::target::{EncoderAdaptersTarget, TrainingTarget};
use jammi_ai::fine_tune::trainer::TrainingLoopBuilder;
use jammi_ai::fine_tune::{EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig, LrSchedule};
use jammi_ai::model::backend::candle::CandleBackend;
use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
use jammi_ai::model::{BackendType, LoadedModel, ModelId, ResolvedModel, TokenizerSource};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::storage::{StorageRegistry, StorageUrl};
use jammi_db::store::ArtifactStore;
use jammi_db::ModelTask;
use jammi_encoders::AnyEncoder;
use jammi_lora::{AdapterConfig, LoraInitMode};

use crate::finetune_step::{attention_arm, sha256_and_len};
use crate::report::{EpochHeldOut, FinetuneRunTier};

/// The fused-vs-ALLOFF arm this run was launched under — CALLER-declared
/// PROVENANCE (see this module's own doc), never derived from a dispatch
/// counter (mirrors [`crate::finetune_step::FinetuneStepParams::expect_kernels_disabled`]'s
/// posture: the tier can VALIDATE what the caller claims against what the
/// process's `JAMMI_KERNELS_DISABLE` actually resolved to, but the value
/// itself is an intent the caller states on the command line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The fused cascade — no kernels forced eager.
    Fused,
    /// `JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused` — both
    /// levers ALLOFF at once (CONTRACT Frame; v2 delta 10 pre-registers a
    /// flash-only / adamw-only TRIAGE arm for a RED result, not built here).
    Alloff,
}

impl Arm {
    pub fn as_str(self) -> &'static str {
        match self {
            Arm::Fused => "fused",
            Arm::Alloff => "alloff",
        }
    }
}

impl std::str::FromStr for Arm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fused" => Ok(Arm::Fused),
            "alloff" => Ok(Arm::Alloff),
            other => Err(format!(
                "--arm '{other}' is invalid: expected 'fused' or 'alloff'"
            )),
        }
    }
}

/// Re-box a non-`Send` error (e.g. [`crate::finetune_step::sha256_and_len`]'s
/// `Box<dyn Error>`) into this module's `Send + Sync` error type — needed
/// because [`run`] is driven from `tokio::task::spawn_blocking` (it calls
/// `Handle::current().block_on(..)` internally for catalog I/O, mirroring
/// `fine_tune::worker::run_fine_tune_blocking`), and `spawn_blocking`
/// requires its future's `Ok`/`Err` to be `Send`.
fn sendify<E: std::fmt::Display>(e: E) -> Box<dyn std::error::Error + Send + Sync> {
    e.to_string().into()
}

/// The op keys the `alloff` arm expects `JAMMI_KERNELS_DISABLE` to name —
/// CONTRACT Frame's `ALLOFF=attention_block_flash,adamw_step_fused` verbatim.
pub const ALLOFF_KEYS: [&str; 2] = ["attention_block_flash", "adamw_step_fused"];

/// Which embedding objective this run trains — CONTRACT H4's 2026-08-28
/// amendment ("objective selection under the triplet-shaped fixture"): H4a
/// found the committed H3 fixture TRIPLET-shaped
/// (`anchor_id\tpositive_id\tnegative_id`), while the Frame's own
/// "embedding_loss+temp" phrasing anticipated MNRL. Both families train over
/// the SAME committed fixture — `Triplet` natively (all three columns),
/// `Mnrl` via the (anchor, positive) PROJECTION of the identical rows in the
/// identical committed order (see [`project_to_pairs`]): dropping the mined
/// negative column and letting the rest of each batch supply in-batch
/// negatives instead. H5 step 0's dynamic-range probe runs BOTH (one seed
/// each) and pre-registers which one v1 keeps; this tier does not itself
/// choose — the CALLER selects via `--objective`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// `EmbeddingLoss::Triplet { margin }` — explicit mined negative per row.
    Triplet,
    /// `EmbeddingLoss::MultipleNegativesRanking { temperature }` — in-batch
    /// negatives only, over the fixture's (anchor, positive) projection.
    Mnrl,
}

impl Objective {
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::Triplet => "triplet",
            Objective::Mnrl => "mnrl",
        }
    }
}

impl std::str::FromStr for Objective {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "triplet" => Ok(Objective::Triplet),
            "mnrl" => Ok(Objective::Mnrl),
            other => Err(format!(
                "--objective '{other}' is invalid: expected 'triplet' or 'mnrl'"
            )),
        }
    }
}

/// Project committed [`IdTriplet`] rows to the MNRL (anchor, positive) shape
/// — the SAME rows, in the SAME committed order, with the mined negative
/// column DROPPED (CONTRACT amendment: "the fixture's (anchor, positive)
/// projection serves MNRL losslessly; in-batch negatives replace the mined
/// negative"). A pure function (no id, no shuffling) so both the train split
/// and the held-out fixture project identically — [`run`] calls this for
/// both when [`Objective::Mnrl`] is selected.
fn project_to_pairs(pairs: &[IdTriplet]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|p| (p.anchor.clone(), p.positive.clone()))
        .collect()
}

/// One (anchor, positive, negative) text triplet, keyed by a stable id — the
/// shape both the train split and the held-out fixture are supplied in,
/// regardless of which [`Objective`] this run trains. The committed held-out
/// fixture (`cookbook/fixtures/finetune_heldout`, CONTRACT H3) mines an
/// EXPLICIT negative per row (`heldout_ids.txt`'s
/// `anchor_id\tpositive_id\tnegative_id` shape); [`Objective::Triplet`]
/// consumes all three columns natively, [`Objective::Mnrl`] consumes only
/// the (anchor, positive) projection ([`project_to_pairs`]) — see
/// [`crate::report::FinetuneRunTier::margin`]'s doc for the CONTRACT-vs-
/// fixture naming note.
#[derive(Debug, Clone)]
pub struct IdTriplet {
    pub id: String,
    pub anchor: String,
    pub positive: String,
    pub negative: String,
}

/// Parameters [`run`] drives the tier off. The CPU-hermetic smoke test and
/// the real pod producer share this one shape — only which rows/checkpoint
/// they pass differs.
#[derive(Debug, Clone)]
pub struct FinetuneRunParams {
    /// Directory holding `config.json` + `model.safetensors` (+ optionally
    /// `tokenizer.json` — REQUIRED for the real `EncoderAdapters` target;
    /// see this module's doc).
    pub model_dir: PathBuf,
    /// The caller-declared arm (see [`Arm`]'s own doc).
    pub arm: Arm,
    /// Training rows (disjoint from `heldout_pairs`; see this module's doc).
    pub train_pairs: Vec<IdTriplet>,
    /// The held-out fixture's rows, in the fixture's COMMITTED order — this
    /// order is scoring identity (CONTRACT H1: "the batch partition IS
    /// identity"), never re-sorted or shuffled by this tier.
    pub heldout_pairs: Vec<IdTriplet>,
    /// sha256 (hex) of the `--train-jsonl` FILE's own raw bytes, MEASURED by
    /// `main.rs::load_train_jsonl` off the file this run actually opened —
    /// never a caller-transcribed digest, and NOT the same quantity as the
    /// committed fixture manifest's own `dataset_sha256` (a Merkle over
    /// per-pair digests, built off-process); see
    /// [`crate::report::FinetuneRunTier::train_pairs_file_sha256`]'s own doc
    /// for why this field carries a distinct name (unit-63 adversarial-audit
    /// finding 5(b)).
    pub train_pairs_file_sha256: String,
    /// sha256 (hex) of the held-out id list's committed content — likewise
    /// caller-supplied (the fixture manifest's own `heldout_ids_sha256`).
    pub heldout_ids_sha256: String,
    /// sha256 (hex) of the `--heldout-jsonl` FILE's own raw bytes, MEASURED
    /// by `main.rs::load_heldout_fixture` off the file this run actually
    /// opened — the held-out TEXT is a total determinant of every per-
    /// example loss `d_i`, so (like `heldout_ids_sha256`) it must be
    /// content-anchored, never merely trusted by filename (unit-63
    /// adversarial-audit finding 5(a)).
    pub heldout_pairs_sha256: String,
    pub seed: u64,
    /// Optimizer/schedule/objective knobs. `early_stopping_patience` MUST be
    /// `10_000` (CONTRACT Frame's never-stops idiom) — [`run`] refuses a
    /// smaller value rather than silently letting an early-stopped run
    /// masquerade as a full-budget one.
    pub epochs: usize,
    pub eval_cadence: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub lr_schedule: LrSchedule,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: usize,
    pub validation_fraction: f64,
    pub early_stopping_patience: usize,
    pub early_stopping_metric: EarlyStoppingMetric,
    pub max_grad_norm: f64,
    /// Which embedding objective this run trains (CONTRACT H4 amendment;
    /// see [`Objective`]'s own doc).
    pub objective: Objective,
    /// The Triplet objective's margin — used to build `EmbeddingLoss::Triplet`
    /// when `objective == Objective::Triplet`; ignored (but still a valid,
    /// caller-supplied value) otherwise.
    pub margin: f64,
    /// The MNRL objective's similarity-scale knob — used to build
    /// `EmbeddingLoss::MultipleNegativesRanking` when
    /// `objective == Objective::Mnrl`; ignored otherwise. `20.0` is the
    /// standard default (`jammi_wire::fine_tune::EmbeddingLoss::MultipleNegativesRanking`'s
    /// own doc: "temperature is the similarity scale; 20.0 is the standard
    /// default" — the same value `jammi_ai::fine_tune::trainer::PAIRWISE_SCALE`
    /// falls back to when no MNRL config is set).
    pub temperature: f64,
    pub matryoshka_dims: Vec<usize>,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub target_modules: Vec<String>,
    pub backbone_dtype: jammi_numerics::ComputePrecision,
    pub max_seq_length: usize,
    /// CALLER-declared premise for the `admission_is_dense` report field
    /// (`--expect-dense`, default `false`, matching the committed fixture's
    /// padded transport) — mirrors `arm`'s declared-vs-resolved posture, not
    /// `expect_kernels_disabled`'s: this tier's real-text path drives
    /// `encode_chunk`'s plain `encoder.forward`, which never reaches
    /// `jammi_encoders::ModernBert::forward_with_lengths`'s dense-vs-padded
    /// fork at all (see [`run`]'s own doc), so there is no live,
    /// process-resolved signal on this tier's admission path to validate the
    /// claim against the way `disabled_ops_requested()` validates
    /// `expect_kernels_disabled`. The value is therefore recorded exactly as
    /// stated, never measured — a downstream merger checks it against the
    /// fixture's own known shape, the same way it checks any other
    /// caller-declared premise leg.
    pub expect_dense: bool,
    /// CUDA ordinal, or `None` for CPU (the CPU-hermetic smoke path).
    pub cuda_device: Option<usize>,
    /// Scratch directory this run's catalog sqlite file, artifact store, and
    /// training-scratch tempdirs live under — the CALLER owns its lifetime
    /// (kept alive for the whole run).
    pub work_dir: PathBuf,

    // ── Mutant provenance (unit 63 round-7 audit, finding 1) ────────────
    //
    // These three are HONEST-LABELING fields, never identity or provenance
    // in [`crate::report::FinetuneRunTier::IDENTITY_FIELDS`]/
    // [`crate::report::FinetuneRunTier::PROVENANCE_FIELDS`]'s sense: a
    // mutant leg is an ordinary `fused`-arm run (same config, same
    // checkpoint, same fixture) with one AdamW-update-scaling patch
    // substituted into the binary this process was compiled from — nothing
    // about the RUN's own identity/provenance tuple differs from a clean
    // `fused` leg (`mutants/README.md`'s "what M1 does NOT touch"), so
    // stamping the mutant's name onto either comparison tuple would make a
    // mutant leg permanently unpairable with the clean legs it exists to be
    // compared against (CONTRACT H4's "the arm is provenance, never
    // identity" logic applies here a fortiori: identity/provenance name
    // WHAT WAS MEASURED, this names WHICH BINARY did the measuring). They
    // are a caller-declared self-report — this process cannot verify from
    // inside itself that it was actually built from the claimed patch —
    // exactly as honest as a person naming themselves, never a measured or
    // derived fact; the downstream `ab_merge.py` mutant-dose-ladder mode
    // reads them by these exact key names to attribute a dose column's legs
    // to a specific, auditable mutant patch (`mutants/README.md`'s own
    // recorded fields).
    //
    // All three are OPTIONAL and all-or-none: [`run`] refuses (typed error,
    // not a panic) unless EITHER all three were never touched (`None`) OR
    // all three are non-empty after trimming — round-8 finding 4 tightened
    // this from a mere `is_some()` presence count to a non-emptiness count,
    // since the CLI happily parses `--mutant-base-sha ""` (and any
    // whitespace-only value) into `Some(_)`; a trio that is explicitly
    // supplied but empty/whitespace in every position is refused too, not
    // silently treated as the ordinary non-mutant case — see [`run_impl`]'s
    // own leading validation block, which also shape-checks
    // `mutant_base_sha` (7-40 hex chars) and `mutant_patch_sha256` (exactly
    // 64 hex chars) once the trio clears the emptiness gate. A normal
    // (non-mutant) leg supplies `None` for all three, and the emitted JSON
    // omits all three keys entirely (`#[serde(skip_serializing_if =
    // "Option::is_none")]` on [`crate::report::FinetuneRunTier`]'s mirror
    // fields), so a normal leg's report bytes are unchanged by this finding
    // (committed goldens unaffected).
    /// `--mutant-id`: the mutant's own label (e.g. `"eps-0.10"` — see
    /// `docs/plans/63-how-well/mutants/README.md`'s dose-family naming).
    /// Trimmed and checked for non-emptiness by `run_impl`; the STAMPED
    /// value (in the returned tier) is the trimmed string.
    pub mutant_id: Option<String>,
    /// `--mutant-base-sha`: the git commit sha this mutant's patch was cut
    /// against. Trimmed and shape-checked (7-40 hex chars) by `run_impl`;
    /// the STAMPED value is the trimmed string.
    pub mutant_base_sha: Option<String>,
    /// `--mutant-patch-sha256`: sha256 (hex) of the mutant patch's own
    /// content — the "auditable" half of "attributable to a specific,
    /// auditable mutant patch". Trimmed and shape-checked (exactly 64 hex
    /// chars) by `run_impl`; the STAMPED value is the trimmed string.
    pub mutant_patch_sha256: Option<String>,
}

/// A held-out example-mean loss point measured after one training epoch —
/// [`crate::report::EpochHeldOut`] is the serialized shape; this pairs it
/// with the model_type dispatch this module needs internally.
struct Trajectory {
    points: Vec<EpochHeldOut>,
}

/// Read a `config.json`'s `model_type`, defaulting to `"bert"` — the SAME
/// default `CandleBackend::load` applies for a checkpoint that omits the
/// field (this tier's dispatch must agree with the base-model loader's own
/// dispatch, or the two would disagree about which architecture a directory
/// names).
fn model_type_of(model_config: &serde_json::Value) -> String {
    model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("bert")
        .to_string()
}

/// Build the frozen base model this run's `EncoderAdapters` target needs —
/// ONLY for its TOKENIZER (`TrainingLoop::encode_texts`'s `EncoderAdapters`
/// branch reads `base.tokenizer`, never `base`'s own forward; see that
/// method's doc). Constructed via the REAL `CandleBackend::load` public
/// trait method — the same backend a serving `InferenceSession` loads
/// through — over a hand-built [`ResolvedModel`] naming this run's
/// `model_dir` directly (mirroring the CLI-flag `--model-dir` convention
/// every other tier in this crate already uses, rather than the catalog-
/// registered `ModelSource` resolution production job submission goes
/// through — that resolution is a job-submission/catalog concern, not
/// something a bench tier that already knows its model path needs to
/// replicate).
fn load_base_model(
    model_dir: &Path,
    device_config: &DeviceConfig,
) -> Result<Arc<LoadedModel>, Box<dyn std::error::Error + Send + Sync>> {
    let config_path = model_dir.join("config.json");
    let model_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "finetune-run: {} has no tokenizer.json — the EncoderAdapters training target \
             requires a real tokenizer to turn the fixture's text pairs into token ids (this \
             tier never falls back to synthetic ids the way finetune-step does, because a \
             synthetic-id run would never touch the committed fixture's real text at all)",
            model_dir.display()
        )
        .into());
    }
    let pooling_config_path = model_dir.join("1_Pooling/config.json");
    let pooling_config = if pooling_config_path.exists() {
        Some(serde_json::from_str(&std::fs::read_to_string(
            &pooling_config_path,
        )?)?)
    } else {
        None
    };
    let resolved = ResolvedModel {
        model_id: ModelId(model_dir.display().to_string()),
        backend: BackendType::Candle,
        task: ModelTask::TextEmbedding,
        config_path,
        weights_paths: vec![model_dir.join("model.safetensors")],
        tokenizer: Some(TokenizerSource::HuggingFaceJson(tokenizer_path)),
        model_config,
        preprocessor_config: None,
        pooling_config,
        base_model_id: None,
        adapter_path: None,
        estimated_memory: 0,
    };
    let backend = CandleBackend;
    Ok(Arc::new(backend.load(&resolved, device_config)?))
}

/// Build a FRESH LoRA-injected encoder over `model_dir`, deterministic from
/// `(seed, target_modules, lora_rank, lora_alpha, lora_dropout)` —
/// [`jammi_lora::LoraInitMode::ZerosB`], the SAME init mode
/// `crate::finetune_step::build_fixture` uses, so two fresh builds at the
/// identical config start from byte-identical LoRA weights (the property
/// this tier's resume-cycling relies on: epoch k+1's fresh build, THEN
/// overwritten by the epoch-k checkpoint restore, never needs the fresh
/// init itself to carry any information — the restore always wins for every
/// registered `Var` name).
#[allow(clippy::too_many_arguments)]
fn build_encoder_adapters(
    model_dir: &Path,
    model_type: &str,
    target_modules: &[String],
    lora_rank: usize,
    lora_alpha: f64,
    lora_dropout: f64,
    backbone_dtype: jammi_numerics::ComputePrecision,
    seed: u64,
    device: &Device,
    varmap: &VarMap,
) -> Result<(AnyEncoder, AdapterConfig), Box<dyn std::error::Error + Send + Sync>> {
    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))?;
    let weights = model_dir.join("model.safetensors");
    let dtype = jammi_encoders::compute_precision_to_dtype(backbone_dtype);
    let empty_ranks: HashMap<String, usize> = HashMap::new();
    let no_layers: Option<Vec<usize>> = None;
    let lora_dropout_opt = (lora_dropout > 0.0).then_some(lora_dropout as f32);
    let lora_build_1 = jammi_lora::LoraBuildConfig {
        target_modules,
        layers_to_transform: &no_layers,
        lora_rank,
        lora_alpha,
        use_rslora: false,
        lora_dropout: lora_dropout_opt,
        rank_pattern: &empty_ranks,
        init_mode: LoraInitMode::ZerosB,
        seed,
    };
    let mut encoder = match model_type {
        "modernbert" => {
            let cfg: jammi_encoders::ModernBertConfig = serde_json::from_str(&config_raw)?;
            let m = jammi_encoders::ModernBert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            AnyEncoder::ModernBert(m)
        }
        "bert" => {
            let cfg: jammi_encoders::BertConfig = serde_json::from_str(&config_raw)?;
            let m = jammi_encoders::Bert::builder()
                .pooling(jammi_encoders::Pooling::Mean)
                .backbone_dtype(dtype)
                .lora(lora_build_1)
                .build(&[weights.as_path()], &cfg, device, varmap)?;
            AnyEncoder::Bert(m)
        }
        other => {
            return Err(format!(
                "finetune-run: unsupported model_type '{other}' — this tier supports 'bert' and \
                 'modernbert' (the C16 gate's checkpoint family and this crate's generic CPU test \
                 fixture)"
            )
            .into())
        }
    };
    // Re-audit round-2 fix (unit 63 finding 2): `ModernBert::builder().build(..)` /
    // `Bert::builder().build(..)` construct a FRESH encoder in EVAL mode
    // (`training: false` at construction — see each builder's own `build`)
    // — a plain `TrainingTarget::EncoderAdapters(..)` wrap of that fresh
    // encoder therefore drives every forward through the EVAL attention
    // path, and `ModernBertAttention::forward`'s `self.training` gate never
    // reaches `forward_training_attention` at all, so the fused
    // whole-attention-block kernel (the C16 A/B's entire measurand) cannot
    // dispatch in EITHER arm — the fused leg is silently indistinguishable
    // from a bug that always runs eval. `TrainingLoop::run()` itself never
    // calls `set_training(true)` for the ordinary (non-GradCache)
    // per-batch path either (only `mine_hard_negatives`/`run_gradcache_epoch`
    // do, and neither applies to `EncoderAdapters`), so this call is NOT
    // redundant with anything the trainer does today — mirrors
    // `finetune_step.rs`'s own `build_fixture`'s `encoder.set_training(true)`
    // call EXACTLY, so this tier's fresh-per-epoch build starts from the
    // same training-mode precondition that tier's fixture always has. This
    // is belt-and-braces alongside `run`'s own dispatch-counter proof below
    // (`attention_block_fused_dispatches`/`attention_block_eager_dispatches`/
    // `attention_block_flash_*`): a caller must not depend on either fix in
    // isolation to catch a future regression in the other.
    encoder.set_training(true);
    // A SECOND `LoraBuildConfig` (identical values) — `lora_build_1` above
    // was already MOVED into `.lora(...)`, and `AdapterConfig::from_build`
    // needs its own borrow; the type is a plain, cheap struct of scalars and
    // borrowed slices, so building it twice is free.
    let lora_build_2 = jammi_lora::LoraBuildConfig {
        target_modules,
        layers_to_transform: &no_layers,
        lora_rank,
        lora_alpha,
        use_rslora: false,
        lora_dropout: lora_dropout_opt,
        rank_pattern: &empty_ranks,
        init_mode: LoraInitMode::ZerosB,
        seed,
    };
    let adapter_cfg = AdapterConfig::from_build(model_type, &lora_build_2, backbone_dtype);
    Ok((encoder, adapter_cfg))
}

/// Build the [`FineTuneConfig`] this run's every epoch leg shares — only
/// `epochs` varies leg to leg (see [`run`]'s resume-cycle).
fn base_config(params: &FinetuneRunParams, epochs: usize) -> FineTuneConfig {
    FineTuneConfig {
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: params.lora_dropout,
        learning_rate: params.learning_rate,
        epochs,
        batch_size: params.batch_size,
        max_seq_length: params.max_seq_length,
        embedding_loss: Some(match params.objective {
            Objective::Triplet => EmbeddingLoss::Triplet {
                margin: params.margin,
            },
            Objective::Mnrl => EmbeddingLoss::MultipleNegativesRanking {
                temperature: params.temperature,
            },
        }),
        classification_loss: None,
        regression_loss: None,
        quantile_levels: Vec::new(),
        gradient_accumulation_steps: params.gradient_accumulation_steps,
        validation_fraction: params.validation_fraction,
        early_stopping_patience: params.early_stopping_patience,
        warmup_steps: params.warmup_steps,
        lr_schedule: params.lr_schedule,
        early_stopping_metric: params.early_stopping_metric,
        target_modules: params.target_modules.clone(),
        layers_to_transform: None,
        use_rslora: false,
        rank_pattern: HashMap::new(),
        init_lora_weights: LoraInitMode::ZerosB,
        backbone_dtype: params.backbone_dtype,
        weight_decay: params.weight_decay,
        max_grad_norm: params.max_grad_norm,
        cached: false,
        hard_negatives: Default::default(),
        matryoshka_dims: params.matryoshka_dims.clone(),
        seed: params.seed,
    }
}

/// Run this tier: `params.epochs` resume-chained single-epoch legs over the
/// REAL `TrainingLoopBuilder`, calling `evaluate_held_out` on the fixture at
/// `eval_cadence` and unconditionally on the last epoch. See this module's
/// own doc for the full design rationale.
pub fn run(
    params: &FinetuneRunParams,
) -> Result<FinetuneRunTier, Box<dyn std::error::Error + Send + Sync>> {
    run_impl(params, true).map(|(tier, _final_varmap)| tier)
}

/// [`run`]'s real body, plus a test-only `probe_at_init` escape hatch and the
/// final epoch's [`VarMap`] handle — NEITHER is reachable from the public
/// `--` CLI surface or [`run`] itself (which always passes `true` and
/// discards the varmap). Exists solely so
/// `tests::init_probe_does_not_perturb_the_training_trajectory_bitwise`
/// (amendment 2026-08-29b, item 4) can drive the identical resume-cycle with
/// and without the new pre-`run()` init probe and compare the RESULT —
/// including the actual trained weights, not merely the reported numbers —
/// bit for bit.
fn run_impl(
    params: &FinetuneRunParams,
    probe_at_init: bool,
) -> Result<(FinetuneRunTier, VarMap), Box<dyn std::error::Error + Send + Sync>> {
    if params.early_stopping_patience < 10_000 {
        return Err(format!(
            "finetune-run: --early-stopping-patience {} is below the CONTRACT Frame's never-\
             stops idiom (10_000) — a run that can early-stop before the configured epoch \
             budget cannot be paired at the FINAL epoch the C16/H2 sign test requires",
            params.early_stopping_patience
        )
        .into());
    }
    if params.epochs == 0 {
        return Err("finetune-run: --epochs 0 has no final epoch to measure".into());
    }
    // Mutant provenance is all-or-none (unit 63 round-7 audit, finding 1;
    // tightened round-8, finding 4): a subset of the three flags present but
    // incomplete is a labeling error the merger could not attribute to a
    // specific patch either way (`finetune_run_mutant_column_violations`'s
    // per-field emptiness check), so this producer refuses it up front
    // rather than emitting a half-labeled leg a downstream reader might
    // mistake for either a clean leg or a fully-attributed mutant one.
    //
    // The invariant is NON-EMPTINESS, not presence: the CLI parses
    // `--mutant-base-sha ""` as `Some(String::new())`, and a
    // whitespace-only value is just as un-attributable as an empty one, so
    // each supplied value is trimmed FIRST. Two, and only two, states clear
    // this gate: (a) NONE of the three flags was ever touched (`None` all
    // the way — the ordinary non-mutant leg), or (b) all three were
    // supplied AND are non-empty-after-trim (a fully, honestly labeled
    // mutant leg). Every other state is refused, INCLUDING a trio that was
    // explicitly supplied but is empty or whitespace-only in every
    // position — that is not the same thing as never touching the flags at
    // all, and stamping it as a clean leg would silently launder a caller
    // mistake into the same bytes an ordinary leg produces. The stamped
    // values (below, and in the JSON `tier` this function returns) are the
    // trimmed strings, never the raw, possibly-padded CLI input.
    let mutant_id = params
        .mutant_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    // unit-63 round-9 audit advisory (b), comment corrected by round-10
    // audit F2: lowercased AFTER trim, not merely `to_string`'d, so the
    // stamped artifact records canonical-case hex (sha is case-insensitive
    // by domain). CANONICALIZATION ONLY -- `mutant_base_sha` has no
    // downstream comparison anywhere in this pair (ab_merge.py only checks
    // it for presence, `finetune_run_mutant_column_violations`'s `for
    // field in (...)` loop), so nothing here depends on this lowercasing;
    // it exists solely so a human reading the artifact sees one consistent
    // case convention.
    let mutant_base_sha = params
        .mutant_base_sha
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase);
    // Same canonicalization, but `mutant_patch_sha256` DOES have a
    // downstream comparison: ab_merge.py's
    // `finetune_run_mutant_column_violations` checks this leg's own
    // stamped value against the caller-supplied `--mutant-legs` spec.
    // Round-9 advisory (b) lowercased ONLY this producer side, which
    // regressed an all-uppercase leg/spec pair (previously an exact-string
    // match) into a false "labeling error" the moment this side started
    // normalizing and the caller side did not. Round-10 audit F2 fixed the
    // comparison ITSELF -- ab_merge.py's `finetune_run_mutant_column_violations`
    // (the per-leg comparison) and the `--mutant-legs` CLI fold (the spec
    // parse) -- to case-fold both sides at the comparison site, so this
    // producer-side lowercasing is back to being canonicalization of the
    // artifact, never something the comparison's correctness depends on.
    // Cited by FUNCTION NAME, never by line number: ab_merge.py's own
    // line numbers have already rotted past this comment once (unit-63
    // round-11 audit advisory (b)).
    let mutant_patch_sha256 = params
        .mutant_patch_sha256
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase);
    let never_touched = params.mutant_id.is_none()
        && params.mutant_base_sha.is_none()
        && params.mutant_patch_sha256.is_none();
    let fully_labeled =
        mutant_id.is_some() && mutant_base_sha.is_some() && mutant_patch_sha256.is_some();
    if !never_touched && !fully_labeled {
        // Names each flag's actual state — `None` (never touched), a
        // trimmed value, or "supplied but empty/whitespace-only" (the
        // finding-4 case the old `is_some()` count silently accepted) —
        // rather than the raw `Option<String>` `Debug` output, so a
        // whitespace-only value doesn't render as an indistinguishable
        // `Some("")`-shaped string in the refusal.
        let describe = |raw: &Option<String>, trimmed: &Option<String>| match (raw, trimmed) {
            (None, _) => "absent".to_string(),
            (Some(_), None) => "supplied but empty-or-whitespace-only".to_string(),
            (Some(_), Some(v)) => format!("{v:?}"),
        };
        return Err(format!(
            "finetune-run: --mutant-id/--mutant-base-sha/--mutant-patch-sha256 are all-or-none \
             (a value that is empty or whitespace-only after trimming is not a real label, but \
             supplying one is also not the same as never touching the flag) — got \
             mutant_id={}, mutant_base_sha={}, mutant_patch_sha256={} (a partial or blank \
             mutant label cannot be attributed to a specific, auditable mutant patch)",
            describe(&params.mutant_id, &mutant_id),
            describe(&params.mutant_base_sha, &mutant_base_sha),
            describe(&params.mutant_patch_sha256, &mutant_patch_sha256),
        )
        .into());
    }
    // Shape validation, only reachable once all three cleared the
    // non-emptiness gate above: a trio that is non-empty but malformed
    // (not a real sha) is just as un-attributable as a partial trio, so it
    // gets the same typed refusal, each naming the offending flag.
    if let (Some(mutant_base_sha), Some(mutant_patch_sha256)) =
        (mutant_base_sha.as_deref(), mutant_patch_sha256.as_deref())
    {
        let is_hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit());
        if !(7..=40).contains(&mutant_base_sha.len()) || !is_hex(mutant_base_sha) {
            return Err(format!(
                "finetune-run: --mutant-base-sha {mutant_base_sha:?} must be 7-40 hex chars (a \
                 git commit sha), got length {} after trim",
                mutant_base_sha.len()
            )
            .into());
        }
        if mutant_patch_sha256.len() != 64 || !is_hex(mutant_patch_sha256) {
            return Err(format!(
                "finetune-run: --mutant-patch-sha256 {mutant_patch_sha256:?} must be exactly 64 \
                 hex chars (a sha256 hex digest), got length {} after trim",
                mutant_patch_sha256.len()
            )
            .into());
        }
    }
    // The `alloff` arm's declared intent must actually be what THIS process's
    // `JAMMI_KERNELS_DISABLE` resolved to — the same "declared vs resolved"
    // hard-error `FinetuneStepParams::expect_kernels_disabled` performs (see
    // that field's doc), turning a dropped/mistyped/unforwarded env var into
    // a failure on the SAME invocation rather than a silently-fused `alloff`
    // leg a downstream merger would misread as the treatment arm. The fused
    // arm makes no such claim (an operator may legitimately run it with
    // OTHER, unrelated op keys disabled), so this check only fires for
    // `Arm::Alloff`.
    if params.arm == Arm::Alloff {
        let mut expected: Vec<String> = ALLOFF_KEYS.iter().map(|s| s.to_string()).collect();
        expected.sort();
        let requested = jammi_kernels::admission::disabled_ops_requested();
        if requested != expected {
            return Err(format!(
                "finetune-run: --arm alloff requires JAMMI_KERNELS_DISABLE to resolve to exactly \
                 {expected:?}, but this process's JAMMI_KERNELS_DISABLE resolved to {requested:?} \
                 — the env var was dropped, mistyped, or not forwarded to this process (INVALID \
                 run, not a datum)"
            )
            .into());
        }
    }
    if params.heldout_pairs.is_empty()
        || !params.heldout_pairs.len().is_multiple_of(params.batch_size)
    {
        return Err(format!(
            "finetune-run: {} held-out pairs is not a nonzero multiple of --batch {} (the seam \
             refuses this too, but failing here names the fixture, not an opaque trainer error)",
            params.heldout_pairs.len(),
            params.batch_size
        )
        .into());
    }

    let device = match params.cuda_device {
        Some(ordinal) => Device::new_cuda(ordinal)?,
        None => Device::Cpu,
    };
    let device_config = DeviceConfig {
        gpu_device: params.cuda_device.map(|o| o as i32).unwrap_or(-1),
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: params.backbone_dtype,
    };

    let (checkpoint_config_sha256, _config_len) =
        sha256_and_len(&params.model_dir.join("config.json")).map_err(sendify)?;
    let (checkpoint_weights_sha256, checkpoint_weights_size_bytes) =
        sha256_and_len(&params.model_dir.join("model.safetensors")).map_err(sendify)?;

    let model_config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        params.model_dir.join("config.json"),
    )?)?;
    let model_type = model_type_of(&model_config);

    // The base model, for its tokenizer only (see `load_base_model`'s doc).
    let base_model_arc = load_base_model(&params.model_dir, &device_config)?;

    // Local, file-backed catalog + artifact store — CPU-hermetic (a sqlite
    // file + a `file://` object-store root under `params.work_dir`), the
    // SAME shape `TrainingLoop`'s own resume tests stand up
    // (`trainer.rs`'s `file_store`/its `resume_loop` helper).
    let catalog_dir = params.work_dir.join("catalog");
    std::fs::create_dir_all(&catalog_dir)?;
    let catalog =
        Arc::new(tokio::runtime::Handle::current().block_on(Catalog::open(&catalog_dir))?);
    let artifact_store_root = params.work_dir.join("artifacts");
    std::fs::create_dir_all(&artifact_store_root)?;
    let artifact_cache = params.work_dir.join("artifact_cache");
    std::fs::create_dir_all(&artifact_cache)?;
    let store_url = StorageUrl::parse(
        artifact_store_root
            .to_str()
            .ok_or("finetune-run: work_dir is not valid UTF-8")?,
    )?;
    let artifact_store = Arc::new(ArtifactStore::with_root(
        store_url,
        StorageRegistry::new(),
        artifact_cache,
    )?);
    let artifact_dir = params.work_dir.join("training");
    std::fs::create_dir_all(&artifact_dir)?;

    let job_id = format!("finetune-run-{}-{}-{}", params.arm.as_str(), params.seed, {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    });
    let worker_id = "finetune-run-worker".to_string();
    let model_row_id = format!("{}::finetune-run", params.model_dir.display());
    // `catalog::model_repo`'s composite primary key is
    // `model_pk(tenant, name, version)` = `"{name}::{version}"` — the value
    // `training_jobs.base_model_id`'s FK actually references (`models.model_id`
    // is that composite key, not the bare `model_id` param this call passes
    // as `name`; see `model_repo::model_pk`'s own doc) — never the raw
    // `RegisterModelParams::model_id` alone.
    let model_catalog_pk = format!("{model_row_id}::1");
    tokio::runtime::Handle::current().block_on(catalog.register_model(RegisterModelParams {
        model_id: &model_row_id,
        version: 1,
        model_type: model_type.as_str(),
        backend: "candle",
        task: ModelTask::TextEmbedding,
        base_model_id: None,
        artifact_path: None,
        config_json: None,
    }))?;
    tokio::runtime::Handle::current().block_on(catalog.create_training_job(
        CreateTrainingJobParams {
            job_id: &job_id,
            base_model_id: &model_catalog_pk,
            training_source: "jammi-bench finetune-run",
            loss_type: match params.objective {
                Objective::Triplet => "triplet",
                Objective::Mnrl => "multiple_negatives_ranking",
            },
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        },
    ))?;
    tokio::runtime::Handle::current()
        .block_on(
            catalog.claim_next_training_job(&worker_id, std::time::Duration::from_secs(3600)),
        )?
        .ok_or("finetune-run: the just-created training job was not claimable")?;

    // `Objective::Triplet` consumes the fixture's (anchor, positive,
    // negative) columns natively; `Objective::Mnrl` consumes only the
    // (anchor, positive) PROJECTION of the SAME rows in the SAME committed
    // order ([`project_to_pairs`]) — CONTRACT amendment 2026-08-28. Both
    // loaders below are built from the identical `params.train_pairs` /
    // `params.heldout_pairs` slices, so the row ORDER (and hence
    // `heldout_ids`' pairing with the loader's rows) is identical regardless
    // of which objective this run trains.
    let train_loader = match params.objective {
        Objective::Triplet => {
            let train_rows: Vec<(String, String, String)> = params
                .train_pairs
                .iter()
                .map(|p| (p.anchor.clone(), p.positive.clone(), p.negative.clone()))
                .collect();
            TrainingDataLoader::from_triplets(train_rows)
        }
        Objective::Mnrl => TrainingDataLoader::from_pairs(project_to_pairs(&params.train_pairs)),
    };

    let heldout_ids: Vec<String> = params.heldout_pairs.iter().map(|p| p.id.clone()).collect();
    let heldout_loader = match params.objective {
        Objective::Triplet => {
            let heldout_rows: Vec<(String, String, String)> = params
                .heldout_pairs
                .iter()
                .map(|p| (p.anchor.clone(), p.positive.clone(), p.negative.clone()))
                .collect();
            TrainingDataLoader::from_triplets(heldout_rows)
        }
        Objective::Mnrl => TrainingDataLoader::from_pairs(project_to_pairs(&params.heldout_pairs)),
    };

    // A fixed, deterministic TRAIN-side probe — one batch's worth of the
    // TRAIN rows (never the held-out fixture), scored through the SAME
    // public seam once BEFORE the first `run()` leg (the UNTRAINED model)
    // and then once after EVERY epoch's `run()` leg — CONTRACT amendment
    // 2026-08-29b, item 1(a)/(b), fixing the prior bug where the baseline
    // was taken AFTER epoch 0 had already trained, silently excluding the
    // largest-learning epoch from the premise window despite the field's
    // own doc claiming "over the run" (no contract string is cited here for
    // that endpoint choice; the amendment above is the pre-registration).
    // This producer emits the result as a RAW per-epoch series
    // (`train_probe_series`: index 0 = the untrained/init probe, one entry
    // per epoch thereafter, last = final) — never a pre-derived scalar; a
    // downstream merger derives the "learning happened" premise from the
    // series (the rule lives where rules live, not on this producer). This
    // is honestly a per-example loss under `evaluate_held_out`'s own
    // batch-partition convention (Triplet's margin loss, or MNRL's
    // batch-coupled in-batch-negative loss — see [`Objective`]'s doc), not
    // the trainer's internal batch-mean `avg_train_loss` (which this tier
    // has no way to read off the public surface — see this module's doc) —
    // labelled as a "probe", never as `avg_train_loss` itself. LoRA init is
    // `ZerosB` (deterministic from `(seed, target_modules)`), so the
    // untrained probe reads a deterministic value and an lr=0 leg's whole
    // series is constant (the floor still bites).
    let probe_len = params.batch_size.min(params.train_pairs.len());
    let probe_pairs = &params.train_pairs[..probe_len];
    if probe_len == 0 || !probe_len.is_multiple_of(params.batch_size) {
        return Err(format!(
            "finetune-run: {} train pairs is fewer than --batch {} — cannot build a train-side \
             learning-happened probe batch",
            params.train_pairs.len(),
            params.batch_size
        )
        .into());
    }
    let probe_ids: Vec<String> = probe_pairs.iter().map(|p| p.id.clone()).collect();
    let probe_triplet_rows: Vec<(String, String, String)> = probe_pairs
        .iter()
        .map(|p| (p.anchor.clone(), p.positive.clone(), p.negative.clone()))
        .collect();
    let probe_pair_rows: Vec<(String, String)> = project_to_pairs(probe_pairs);

    let mut trajectory = Trajectory { points: Vec::new() };
    // Amendment 2026-08-29b: the raw probe series, index 0 = the untrained
    // model's init probe, one entry per epoch thereafter (see the doc above
    // on `probe_len`).
    let mut train_probe_series: Vec<f64> = Vec::with_capacity(params.epochs + 1);
    let mut cumulative_steps = 0usize;
    let mut last_final_loss = 0.0f64;
    let mut last_held_out = None;
    // Test-only (see `run_impl`'s own doc): the final epoch's `VarMap`
    // handle — an `Arc`-shared clone taken fresh each epoch, so the LAST
    // clone (after the loop) always points at the trained weights the final
    // epoch's `run()` leg produced.
    let mut last_varmap: Option<VarMap> = None;

    // Fused-dispatch-proof channel (unit 63 re-audit round-2 finding 2):
    // mirrors `finetune_step.rs::run`'s "before"/"after" dispatch-counter
    // snapshot convention EXACTLY (same functions, same field names on the
    // emitted tier — see `FinetuneRunTier`'s own field docs). Taken once
    // around the WHOLE `epochs`-long resume-cycle below (not per epoch):
    // this tier's counters describe "one full (seed, arm) fine-tune run",
    // the same scope every other field on this tier is reported over, and
    // every `training_loop.run(..)`/`evaluate_held_out(..)` call in the
    // loop below (train steps, held-out eval, and the train-side probe)
    // shares the SAME process-wide counters, so a single before/after pair
    // here already covers all of them without double-counting or gaps.
    let ln_dispatch_before = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_before = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_before = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_before = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_before = jammi_lora::lora_epilogue_dispatch_snapshot();
    let lora_linear_fused_dispatch_before = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_before = jammi_encoders::attention_block_dispatch_snapshot();
    let adamw_dispatch_before =
        jammi_kernels::admission::counters_for("adamw_step_fused").snapshot();
    let attention_block_flash_dispatch_before =
        jammi_encoders::attention_block_flash_dispatch_snapshot();

    for epoch_idx in 0..params.epochs {
        let varmap = VarMap::new();
        // Test-only capture (see `run_impl`'s own doc): an `Arc`-shared
        // clone of this epoch's `VarMap`, taken BEFORE it is moved into
        // `TrainingLoopBuilder::new` below — `VarMap::clone` clones the
        // `Arc<Mutex<HashMap<..>>>` pointer, never the tensors themselves,
        // so this clone keeps observing the SAME `Var`s the optimizer
        // mutates in place for the rest of this epoch's `run()` leg.
        // Overwritten every epoch, so after the loop it names the FINAL
        // epoch's trained weights.
        last_varmap = Some(varmap.clone());
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &params.model_dir,
            &model_type,
            &params.target_modules,
            params.lora_rank,
            params.lora_alpha,
            params.lora_dropout,
            params.backbone_dtype,
            params.seed,
            &device,
            &varmap,
        )?;
        let target = TrainingTarget::EncoderAdapters(Box::new(EncoderAdaptersTarget {
            encoder,
            adapter_cfg,
        }));
        let config = base_config(params, epoch_idx + 1);

        let resume = if epoch_idx == 0 {
            None
        } else {
            let fetched = tokio::runtime::Handle::current()
                .block_on(artifact_store.fetch_resume_checkpoint(&job_id))?
                .ok_or_else(|| {
                    format!(
                        "finetune-run: no durable resume checkpoint found for job {job_id} \
                         after epoch {} — the trainer's own epoch-boundary save \
                         (`save_resume_checkpoint`) must have run on every prior epoch",
                        epoch_idx - 1
                    )
                })?;
            Some(load_bundle(fetched.dir(), &device)?)
        };

        let mut builder = TrainingLoopBuilder::new(target, varmap, config)
            .base_model(Arc::clone(&base_model_arc))
            .job_id(job_id.clone())
            .worker_id(worker_id.clone())
            .catalog(Arc::clone(&catalog))
            .artifact_dir(artifact_dir.clone())
            .device(device.clone())
            .cancel(Arc::new(AtomicBool::new(false)))
            .artifact_store(Arc::clone(&artifact_store));
        if let Some(restored) = resume {
            builder = builder.resume(restored);
        }
        let mut training_loop = builder.build()?;

        if epoch_idx == 0 && probe_at_init {
            // Amendment 2026-08-29b, item 1(a): anchor the series at the
            // UNTRAINED model — one `evaluate_held_out` call on the
            // train-probe batch BEFORE this run's first `run()` leg (LoRA
            // init is `ZerosB`, so this is deterministic from `(seed,
            // target_modules)` alone). `probe_at_init` is test-only (see
            // `run_impl`'s own doc) — [`run`] always passes `true`.
            let probe_loader = match params.objective {
                Objective::Triplet => TrainingDataLoader::from_triplets(probe_triplet_rows.clone()),
                Objective::Mnrl => TrainingDataLoader::from_pairs(probe_pair_rows.clone()),
            };
            let init_probe = training_loop.evaluate_held_out(&probe_loader, &probe_ids)?;
            train_probe_series.push(init_probe.mean);
        }

        let result = training_loop.run(&train_loader)?;
        cumulative_steps += result.total_steps;
        last_final_loss = result.final_loss;

        let is_final = epoch_idx + 1 == params.epochs;
        let due = params.eval_cadence > 0 && (epoch_idx + 1).is_multiple_of(params.eval_cadence);
        if due || is_final {
            let held_out = training_loop.evaluate_held_out(&heldout_loader, &heldout_ids)?;
            trajectory.points.push(EpochHeldOut {
                epoch: epoch_idx,
                held_out_mean: held_out.mean,
                held_out_tie_fraction: held_out.tie_fraction,
                held_out_batch_partition_sha256: held_out.batch_partition_sha256.clone(),
            });
            last_held_out = Some(held_out);
        }

        // Amendment 2026-08-29b, item 1(b): probe EVERY epoch (never only
        // the first/final) — the producer emits the RAW series, a
        // downstream merger derives the "learning happened" premise from
        // it (`init_probe - final_probe > floor`).
        let probe_loader = match params.objective {
            Objective::Triplet => TrainingDataLoader::from_triplets(probe_triplet_rows.clone()),
            Objective::Mnrl => TrainingDataLoader::from_pairs(probe_pair_rows.clone()),
        };
        let probe = training_loop.evaluate_held_out(&probe_loader, &probe_ids)?;
        train_probe_series.push(probe.mean);
    }

    // "After" half of the before/after pair taken above the loop — same
    // mechanism, same field names `finetune_step.rs::run` emits.
    let ln_dispatch_after = jammi_encoders::ln_dispatch_snapshot();
    let rope_dispatch_after = jammi_encoders::rope_dispatch_snapshot();
    let softmax_dispatch_after = jammi_encoders::softmax_dispatch_snapshot();
    let geglu_dispatch_after = jammi_encoders::geglu_dispatch_snapshot();
    let lora_epilogue_dispatch_after = jammi_lora::lora_epilogue_dispatch_snapshot();
    let lora_linear_fused_dispatch_after = jammi_lora::lora_linear_fused_dispatch_snapshot();
    let attention_block_dispatch_after = jammi_encoders::attention_block_dispatch_snapshot();
    let adamw_dispatch_after =
        jammi_kernels::admission::counters_for("adamw_step_fused").snapshot();
    let attention_block_flash_dispatch_after =
        jammi_encoders::attention_block_flash_dispatch_snapshot();

    let ln_fused_dispatches = ln_dispatch_after
        .fused
        .saturating_sub(ln_dispatch_before.fused);
    let ln_eager_dispatches = ln_dispatch_after
        .eager
        .saturating_sub(ln_dispatch_before.eager);
    let rope_fused_dispatches = rope_dispatch_after
        .fused
        .saturating_sub(rope_dispatch_before.fused);
    let rope_eager_dispatches = rope_dispatch_after
        .eager
        .saturating_sub(rope_dispatch_before.eager);
    let softmax_fused_dispatches = softmax_dispatch_after
        .fused
        .saturating_sub(softmax_dispatch_before.fused);
    let softmax_eager_dispatches = softmax_dispatch_after
        .eager
        .saturating_sub(softmax_dispatch_before.eager);
    let geglu_fused_dispatches = geglu_dispatch_after
        .fused
        .saturating_sub(geglu_dispatch_before.fused);
    let geglu_eager_dispatches = geglu_dispatch_after
        .eager
        .saturating_sub(geglu_dispatch_before.eager);
    let lora_epilogue_fused_dispatches = lora_epilogue_dispatch_after
        .fused
        .saturating_sub(lora_epilogue_dispatch_before.fused);
    let lora_epilogue_eager_dispatches = lora_epilogue_dispatch_after
        .eager
        .saturating_sub(lora_epilogue_dispatch_before.eager);
    let lora_linear_fused_dispatches = lora_linear_fused_dispatch_after
        .fused
        .saturating_sub(lora_linear_fused_dispatch_before.fused);
    let lora_linear_eager_dispatches = lora_linear_fused_dispatch_after
        .eager
        .saturating_sub(lora_linear_fused_dispatch_before.eager);
    let attention_block_fused_dispatches = attention_block_dispatch_after
        .fused
        .saturating_sub(attention_block_dispatch_before.fused);
    let attention_block_eager_dispatches = attention_block_dispatch_after
        .eager
        .saturating_sub(attention_block_dispatch_before.eager);
    let adamw_fused_dispatches = adamw_dispatch_after
        .fused
        .saturating_sub(adamw_dispatch_before.fused);
    let adamw_eager_dispatches = adamw_dispatch_after
        .eager
        .saturating_sub(adamw_dispatch_before.eager);
    let attention_block_flash_fused_dispatches = attention_block_flash_dispatch_after
        .fused
        .saturating_sub(attention_block_flash_dispatch_before.fused);
    let attention_block_flash_declined_dispatches = attention_block_flash_dispatch_after
        .declined
        .saturating_sub(attention_block_flash_dispatch_before.declined);

    // Belt-and-braces typed refusal (unit 63 re-audit round-2 finding 2):
    // ModernBert is the ONLY architecture with a fused whole-attention-block
    // kernel at all (`bert.rs`'s own `set_training` never touches an
    // attention-block admission path — see that module's doc; a `bert`-arch
    // leg legitimately reads all four counters below as `0` forever, which
    // is why this gate is scoped to `model_type == "modernbert"` and never
    // fires for this crate's generic CPU smoke fixture). For a ModernBert
    // leg that took at least one optimizer step, EVERY training-mode
    // attention forward calls `admit` on exactly one of the block or flash
    // cascade (`ModernBertAttention::forward`'s `self.training` branch,
    // `forward_training_attention`'s doc) — so all four dispatch counters
    // reading zero at once is not a legitimate "declined by domain" outcome
    // (that reads `N eager / 0 fused`, never `0/0/0/0`), it is proof the
    // encoder never entered training mode at all (this finding's root
    // cause: a fresh `ModernBert::builder().build(..)` starts `training:
    // false`, and neither `build_encoder_adapters` above nor
    // `TrainingLoop::run`'s ordinary per-batch path used to flip it).
    // Refusing loudly here beats silently emitting a plausible-looking
    // report for a run that measured the eval path — this check does NOT
    // depend on `encoder.set_training(true)` above (or any trainer-side
    // fix) being correct: it reads the same counters a downstream merger's
    // fused-proof gate reads, independent of how this process got there.
    if model_type == "modernbert"
        && cumulative_steps > 0
        && attention_block_fused_dispatches == 0
        && attention_block_eager_dispatches == 0
        && attention_block_flash_fused_dispatches == 0
        && attention_block_flash_declined_dispatches == 0
    {
        return Err(format!(
            "finetune-run: fused-dispatch-proof failure — this ModernBert run took \
             {cumulative_steps} optimizer step(s) but the training-mode attention path never \
             dispatched in either arm (attention_block_fused_dispatches, \
             attention_block_eager_dispatches, attention_block_flash_fused_dispatches, and \
             attention_block_flash_declined_dispatches are all 0). The encoder was never in \
             training mode, so this run measured the eval path, not the fine-tune step this \
             tier claims to measure — INVALID run, not a datum."
        )
        .into());
    }

    let held_out = last_held_out
        .ok_or("finetune-run: internal: no evaluate_held_out call landed on the final epoch")?;
    // Amendment 2026-08-29b: one probe per epoch, always, plus the init
    // probe when `probe_at_init` is set (only [`run`]'s production path
    // ever sets it `false` — never; that escape hatch is test-only, see
    // `run_impl`'s own doc) — an internal invariant of the loop above, not
    // a caller-triggerable refusal (a wrong count here is this producer's
    // own bug, not a bad input).
    let expected_series_len = params.epochs + usize::from(probe_at_init);
    assert_eq!(
        train_probe_series.len(),
        expected_series_len,
        "finetune-run: internal: train_probe_series must carry the init probe (when requested) \
         plus one entry per epoch"
    );

    let kernels_disabled_requested = jammi_kernels::admission::disabled_ops_requested();
    let kernels_disabled_fired = jammi_kernels::admission::disabled_ops_fired();
    let resolved_attention_arm = attention_arm(&kernels_disabled_requested).to_string();

    // A DECLARED premise, not a measurement: this tier's real-text path
    // never calls `forward_with_lengths` at all (`encode_chunk`'s plain
    // `encoder.forward` never routes through the dense-vs-padded fork
    // `finetune_step.rs`'s `--row-lengths` leg exercises), so there is no
    // live `jammi_kernels::admission`/`jammi_encoders::CompactedBatch`
    // signal on THIS tier's forward path to read back and check the caller
    // against — unlike `kernels_disabled_requested`, which reads a real
    // process-resolved env-var state. `params.expect_dense` is therefore
    // recorded verbatim (CALLER-declared, default `false` matching the
    // committed fixture's padded transport) so a downstream merger's
    // conjunctive premise leg has a concrete, honestly-scoped, checkable
    // fact rather than an inferred one — see `FinetuneRunParams::expect_dense`'s
    // own doc.
    let admission_is_dense = params.expect_dense;

    let max_grad_norm = (params.max_grad_norm > 0.0).then_some(params.max_grad_norm);

    let tier = FinetuneRunTier {
        seed: params.seed,
        batch: params.batch_size,
        seq: params.max_seq_length,
        lora_rank: params.lora_rank,
        lora_alpha: params.lora_alpha,
        lora_dropout: params.lora_dropout,
        margin: match params.objective {
            Objective::Triplet => Some(params.margin),
            Objective::Mnrl => None,
        },
        target_modules: params.target_modules.clone(),
        backbone_dtype: format!("{:?}", params.backbone_dtype).to_lowercase(),
        checkpoint_config_sha256,
        checkpoint_weights_sha256,
        checkpoint_weights_size_bytes,
        max_grad_norm,
        warmup: None,
        row_lengths: None,
        epochs: params.epochs,
        lr: params.learning_rate,
        schedule: format!("{:?}", params.lr_schedule).to_lowercase(),
        warmup_steps: params.warmup_steps,
        weight_decay: params.weight_decay,
        grad_accum: params.gradient_accumulation_steps,
        validation_fraction: params.validation_fraction,
        train_pairs_file_sha256: params.train_pairs_file_sha256.clone(),
        heldout_ids_sha256: params.heldout_ids_sha256.clone(),
        heldout_pairs_sha256: params.heldout_pairs_sha256.clone(),
        heldout_batch_partition_sha256: held_out.batch_partition_sha256.clone(),
        embedding_loss: params.objective.as_str().to_string(),
        temperature: match params.objective {
            Objective::Triplet => None,
            Objective::Mnrl => Some(params.temperature),
        },
        matryoshka_dims: params.matryoshka_dims.clone(),
        early_stopping_patience: params.early_stopping_patience,
        early_stopping_metric: match params.early_stopping_metric {
            EarlyStoppingMetric::TrainLoss => "train_loss".to_string(),
            EarlyStoppingMetric::ValLoss => "val_loss".to_string(),
        },
        eval_cadence: params.eval_cadence,

        arm: params.arm.as_str().to_string(),
        device_name: crate::finetune_step::device_name(params.cuda_device),
        kernels_disabled_requested,
        kernels_disabled_fired,
        flash_compiled: jammi_kernels::admission::FLASH_COMPILED,
        build_features: crate::report::build_features(),
        attention_arm: resolved_attention_arm,
        split_rule: "positional_fraction_split".to_string(),
        batched_forward: true,
        steps_measured: cumulative_steps,

        ln_fused_dispatches,
        ln_eager_dispatches,
        rope_fused_dispatches,
        rope_eager_dispatches,
        softmax_fused_dispatches,
        softmax_eager_dispatches,
        geglu_fused_dispatches,
        geglu_eager_dispatches,
        lora_epilogue_fused_dispatches,
        lora_epilogue_eager_dispatches,
        lora_linear_fused_dispatches,
        lora_linear_eager_dispatches,
        attention_block_fused_dispatches,
        attention_block_eager_dispatches,
        adamw_fused_dispatches,
        adamw_eager_dispatches,
        attention_block_flash_fused_dispatches,
        attention_block_flash_declined_dispatches,

        admission_is_dense,
        tie_fraction: held_out.tie_fraction,

        final_epoch: params.epochs - 1,
        held_out_example_mean: held_out.mean,
        held_out_count: held_out.count,
        final_loss_diagnostic: last_final_loss,
        trajectory: trajectory.points,
        train_probe_series,
        mutant_id,
        mutant_base_sha,
        mutant_patch_sha256,
    };

    let value = serde_json::to_value(&tier).expect("serialize FinetuneRunTier for self-check");
    crate::report::assert_identity_fields_present(&value, FinetuneRunTier::IDENTITY_FIELDS);
    crate::report::assert_identity_fields_present(&value, FinetuneRunTier::PROVENANCE_FIELDS);
    let final_varmap = last_varmap
        .ok_or("finetune-run: internal: no epoch ran, so no final VarMap was captured")?;
    Ok((tier, final_varmap))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cookbook/fixtures/tiny_bert` — the SAME generic, committed fixture
    /// `finetune_run_smoke.rs` drives via the compiled CLI (BERT
    /// architecture, real tokenizer, no consumer shape), resolved relative
    /// to this crate's own manifest dir so this IN-PROCESS test needs no
    /// extra dev-dependency.
    fn tiny_bert_model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../cookbook/fixtures/tiny_bert")
    }

    /// Build the tiny, CPU-hermetic, deterministic [`FinetuneRunParams`] the
    /// non-perturbation test below drives — 4 synthetic train triplets (2
    /// batches at `batch_size 2`), 2 held-out triplets (1 batch), 2 epochs,
    /// `eval_cadence 1`, a nonzero learning rate (so training actually moves
    /// the weights) and a fixed `seed` — two calls built from this same
    /// function differ ONLY in which `work_dir` they own and (via the
    /// caller) `probe_at_init`.
    fn non_perturbation_test_params(work_dir: PathBuf) -> FinetuneRunParams {
        let mk = |offset: usize, n: usize| -> Vec<IdTriplet> {
            (0..n)
                .map(|i| {
                    let k = offset + i;
                    IdTriplet {
                        id: format!("row-{k}"),
                        anchor: format!("synthetic anchor sentence number {k} about widgets"),
                        positive: format!(
                            "synthetic positive sentence number {k} about widgets too"
                        ),
                        negative: format!(
                            "synthetic negative sentence number {k} about gadgets instead"
                        ),
                    }
                })
                .collect()
        };
        FinetuneRunParams {
            model_dir: tiny_bert_model_dir(),
            arm: Arm::Fused,
            train_pairs: mk(0, 4),
            heldout_pairs: mk(100, 2),
            train_pairs_file_sha256: "0".repeat(64),
            heldout_ids_sha256: "1".repeat(64),
            heldout_pairs_sha256: "2".repeat(64),
            seed: 7,
            epochs: 2,
            eval_cadence: 1,
            batch_size: 2,
            learning_rate: 0.01,
            lr_schedule: LrSchedule::Constant,
            warmup_steps: 0,
            weight_decay: 0.0,
            gradient_accumulation_steps: 1,
            // Audit round 7, finding 5: matches the campaign's own
            // `--validation-fraction` default (`FinetuneRunArgs`'s
            // `default_value_t = 0.1`), not an arbitrary `0.0` — with 4 rows
            // this still rounds to a 0-row internal val split
            // (`round(4 * 0.1) == 0`), which is harmless here because
            // `early_stopping_metric` is `TrainLoss` (the `ValLoss`-only
            // empty-loader refusal in `TrainingLoop::run` never fires for
            // this metric), so this change exercises the campaign's real
            // knob value without altering what the resume-cycle actually
            // trains over.
            validation_fraction: 0.1,
            early_stopping_patience: 10_000,
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            max_grad_norm: 0.0,
            objective: Objective::Triplet,
            margin: 0.3,
            temperature: 20.0,
            matryoshka_dims: Vec::new(),
            lora_rank: 2,
            lora_alpha: 4.0,
            // Audit round 7, finding 5: `0.0` made `jammi_lora::LoraLinear`'s
            // `dropout_masks` field structurally `None` for every LoRA layer
            // (`build_encoder_adapters`'s `lora_dropout_opt = (lora_dropout
            // > 0.0).then_some(..)` — `0.0` maps to `None`), so the mask
            // channel `init_probe_does_not_perturb_..._bitwise`'s own doc
            // names ("draws no dropout mask and so touches no RNG stream")
            // was ABSENT, not merely idle: deleting the
            // `with_dropout_disabled` bracket entirely could not have turned
            // this test red, because there was no dropout stream left for a
            // broken bracket to leave un-disabled. `0.05` matches the
            // campaign's own `--lora-dropout` default
            // (`FinetuneRunArgs`'s `default_value_t = 0.05`), making the
            // channel live — see
            // `dropout_forward_counter_is_live_at_the_campaigns_lora_dropout_and_held_still_under_eval_mode`
            // below for the committed proof that the channel is now Some and
            // that toggling training mode is what actually gates it (the
            // RED-provable mechanism `with_dropout_disabled` relies on).
            lora_dropout: 0.05,
            target_modules: vec!["query".to_string(), "value".to_string()],
            backbone_dtype: jammi_numerics::ComputePrecision::F32,
            max_seq_length: 16,
            expect_dense: false,
            cuda_device: None,
            work_dir,
            mutant_id: None,
            mutant_base_sha: None,
            mutant_patch_sha256: None,
        }
    }

    /// The mask-counter proof unit-63 round-7 audit finding 5 requires:
    /// with the campaign's own `lora_dropout` (`0.05`, matching
    /// [`non_perturbation_test_params`]'s now-live value — see that
    /// function's own doc), every LoRA-wrapped layer's dropout forward
    /// counter (`jammi_lora::LoraLinear::dropout_position`,
    /// `AnyEncoder::dropout_positions`) is `Some` (the mask channel
    /// `init_probe_does_not_perturb_the_training_trajectory_bitwise`'s own
    /// doc names — "draws no dropout mask and so touches no RNG stream" —
    /// is actually PRESENT here, not structurally absent the way it was at
    /// `lora_dropout: 0.0`), that it ADVANCES on a training-mode forward
    /// (the state the extra init probe would leave the encoder in if
    /// `TrainingLoop::with_dropout_disabled`'s bracket were bypassed or
    /// removed), and that it HOLDS STILL when training mode is off first —
    /// exactly the `set_training(false)` / call / `set_training(true)`
    /// sequence that bracket performs (`jammi-ai/src/fine_tune/trainer.rs`'s
    /// own doc on `with_dropout_disabled`) — around every
    /// `evaluate_held_out` call.
    ///
    /// This is the "assert via the mask-counter state" form the audit named
    /// as an acceptable alternative to a test-only shadow bypass of
    /// `jammi-ai`'s bracket (this crate does not own `jammi-ai`, so it
    /// cannot commit a mutation there): it proves the property
    /// [`init_probe_does_not_perturb_the_training_trajectory_bitwise`] pins
    /// is now LIVE (can fail), rather than vacuously true, by directly
    /// exhibiting one code path (training-mode forward) that DOES perturb
    /// the counter and one (eval-mode forward) that provably does not — the
    /// exact dichotomy a broken bracket would erase.
    ///
    /// RED proof performed by hand while authoring this fix (not
    /// committed — `jammi-ai` is not this crate's file to modify or commit
    /// against): temporarily editing `TrainingLoop::with_dropout_disabled`
    /// in `jammi-ai/src/fine_tune/trainer.rs` to skip the
    /// `self.set_training(false)` call entirely (leaving only `let
    /// was_training = self.training_mode; let result = f(self);
    /// self.set_training(was_training); result`) and re-running
    /// `init_probe_does_not_perturb_the_training_trajectory_bitwise` at
    /// this fix's now-live `lora_dropout: 0.05` made THAT test fail —
    /// `named_with != named_without` ("trained weights diverged bit-for-bit
    /// between WITH and WITHOUT the init probe") — because the extra init
    /// probe's now-live dropout draw perturbed the first epoch's own
    /// dropout-stream position, changing its trained weights. Re-running
    /// the SAME test at the pre-fix `lora_dropout: 0.0` with the identical
    /// bypass produced NO failure (`named_with == named_without` still
    /// held), confirming finding 5's diagnosis: the pin could not
    /// previously fail because the channel it exists to protect was
    /// structurally absent, not because the bracket was correct. The
    /// `jammi-ai` edit was reverted immediately after both runs (`git diff`
    /// clean on that crate).
    #[test]
    fn dropout_forward_counter_is_live_at_the_campaigns_lora_dropout_and_held_still_under_eval_mode(
    ) {
        let varmap = VarMap::new();
        let (mut encoder, _adapter_cfg) = build_encoder_adapters(
            &tiny_bert_model_dir(),
            "bert",
            &["query".to_string(), "value".to_string()],
            2,
            4.0,
            0.05,
            jammi_numerics::ComputePrecision::F32,
            7,
            &Device::Cpu,
            &varmap,
        )
        .expect("build encoder adapters at lora_dropout 0.05");

        let positions_at_build = encoder
            .dropout_positions()
            .expect("dropout_positions at build");
        assert!(
            !positions_at_build.is_empty(),
            "lora_dropout 0.05 must leave at least one LoRA layer's dropout_masks Some (the \
             channel must be structurally present, unlike the pre-fix lora_dropout 0.0 config): \
             {positions_at_build:?}"
        );
        assert!(
            positions_at_build.values().all(|&p| p == 0),
            "a freshly built encoder must start every dropout counter at 0: \
             {positions_at_build:?}"
        );

        let input_ids = crate::finetune_step::synthetic_ids(2, 4, 256, 99, &Device::Cpu);
        let mask = candle_core::Tensor::ones((2, 4), candle_core::DType::U32, &Device::Cpu)
            .expect("mask tensor");

        // Training-mode forward — the state the extra probe call would run
        // in if `with_dropout_disabled`'s bracket were bypassed: the
        // counter must advance.
        encoder.set_training(true);
        encoder
            .forward(&input_ids, &mask)
            .expect("training-mode forward");
        let positions_after_hot = encoder
            .dropout_positions()
            .expect("dropout_positions after hot forward");
        assert_ne!(
            positions_at_build, positions_after_hot,
            "a training-mode forward must advance the dropout forward counter — this is what \
             proves the channel is genuinely live, and that a bypassed with_dropout_disabled \
             bracket (which would leave the encoder in exactly this training=true state during \
             evaluate_held_out) WOULD perturb it"
        );

        // Eval-mode forward — exactly what `encoder.set_training(false)`
        // leaves the encoder in for the duration of
        // `TrainingLoop::with_dropout_disabled`'s bracket: the counter must
        // hold still.
        encoder.set_training(false);
        let positions_before_cold = encoder
            .dropout_positions()
            .expect("dropout_positions before cold forward");
        encoder
            .forward(&input_ids, &mask)
            .expect("eval-mode forward");
        let positions_after_cold = encoder
            .dropout_positions()
            .expect("dropout_positions after cold forward");
        assert_eq!(
            positions_before_cold, positions_after_cold,
            "an eval-mode (training=false) forward must NOT advance the dropout forward counter \
             — this is the exact mechanism with_dropout_disabled relies on to make the probe \
             read-only"
        );
    }

    /// Flatten every named `Var` in `varmap` to an f32 vector, keyed by name
    /// in a [`std::collections::BTreeMap`] (canonical order — `VarMap`'s own
    /// storage is a plain, iteration-order-unstable `HashMap`) — the
    /// bit-for-bit comparable shape the non-perturbation test below diffs.
    fn named_flat_f32(varmap: &VarMap) -> std::collections::BTreeMap<String, Vec<f32>> {
        let guard = varmap.data().lock().expect("varmap mutex poisoned");
        guard
            .iter()
            .map(|(name, var)| {
                let flat = var
                    .as_tensor()
                    .flatten_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::F32))
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_else(|e| panic!("flatten trained var {name}: {e}"));
                (name.clone(), flat)
            })
            .collect()
    }

    /// CONTRACT amendment 2026-08-29b, item 4 (falsifiability of the "the
    /// corrected probe does not touch the training path" prediction, item
    /// 2(i)): an EXTRA `evaluate_held_out` call on the UNTRAINED model, made
    /// before the very first `run()` leg, must not perturb the resulting
    /// training trajectory at all — `TrainingLoop::evaluate_held_out`'s own
    /// `with_dropout_disabled` bracket ("Dropout bracket" in that method's
    /// doc) is what makes this true: a read-only forward pass with dropout
    /// off draws no dropout mask and so touches no RNG stream the
    /// subsequent `run()` legs consume (see also `no_rng_perturbation`,
    /// `jammi_ai::fine_tune::trainer`'s own seam-level pin of this same
    /// property). Proven here by driving the REAL resume-cycle twice from
    /// the identical seed/config — once WITH the init probe (`run`'s own,
    /// always-on production path) and once WITHOUT it (`probe_at_init:
    /// false`, reachable only via `run_impl`, never from the CLI) — and
    /// asserting the two runs' FINAL TRAINED WEIGHTS are bitwise identical,
    /// not merely their reported summary numbers. CPU-hermetic, over the
    /// tiny generic `tiny_bert` fixture.
    ///
    /// RED proof (performed by hand while authoring this test, not
    /// committed): temporarily changing the per-epoch probe's loader to
    /// re-use `train_loader` instead of a fresh `probe_loader` built from
    /// `probe_triplet_rows`/`probe_pair_rows` (a stand-in for a
    /// perturbation the seam is NOT supposed to have) made this test FAIL —
    /// `named_with != named_without` — confirming the assertion is live,
    /// not vacuously true because both sides always match regardless of
    /// what `run_impl` actually does.
    #[tokio::test]
    async fn init_probe_does_not_perturb_the_training_trajectory_bitwise() {
        let work_dir_with = tempfile::tempdir().expect("tempdir with");
        let work_dir_without = tempfile::tempdir().expect("tempdir without");
        let params_with = non_perturbation_test_params(work_dir_with.path().to_path_buf());
        let params_without = non_perturbation_test_params(work_dir_without.path().to_path_buf());

        let (tier_with, varmap_with) =
            tokio::task::spawn_blocking(move || run_impl(&params_with, true))
                .await
                .expect("join with-probe task")
                .expect("finetune-run WITH the init probe");
        let (tier_without, varmap_without) =
            tokio::task::spawn_blocking(move || run_impl(&params_without, false))
                .await
                .expect("join without-probe task")
                .expect("finetune-run WITHOUT the init probe");

        // `train_probe_series`: WITH carries one extra LEADING entry (the
        // init probe); every entry AFTER that must be bitwise identical to
        // WITHOUT's full (un-prefixed) series — the per-epoch probes
        // themselves must not have been perturbed by the earlier extra
        // call.
        assert_eq!(
            tier_with.train_probe_series.len(),
            tier_without.train_probe_series.len() + 1,
            "WITH must carry exactly one more entry (the init probe) than WITHOUT: {:?} vs {:?}",
            tier_with.train_probe_series,
            tier_without.train_probe_series
        );
        assert_eq!(
            &tier_with.train_probe_series[1..],
            &tier_without.train_probe_series[..],
            "the per-epoch probes diverged once the init probe was added — the seam perturbed \
             the training path"
        );

        // The reported endpoints must match bit for bit.
        assert_eq!(
            tier_with.held_out_example_mean,
            tier_without.held_out_example_mean
        );
        assert_eq!(
            tier_with.final_loss_diagnostic,
            tier_without.final_loss_diagnostic
        );
        assert_eq!(tier_with.steps_measured, tier_without.steps_measured);

        // The strongest form of the claim: the actual TRAINED WEIGHTS, not
        // just the numbers this tier happens to report about them.
        let named_with = named_flat_f32(&varmap_with);
        let named_without = named_flat_f32(&varmap_without);
        assert_eq!(
            named_with, named_without,
            "trained weights diverged bit-for-bit between WITH and WITHOUT the init probe"
        );
    }

    #[test]
    fn objective_from_str_round_trips_both_variants() {
        assert_eq!("triplet".parse::<Objective>().unwrap(), Objective::Triplet);
        assert_eq!("mnrl".parse::<Objective>().unwrap(), Objective::Mnrl);
        assert_eq!(Objective::Triplet.as_str(), "triplet");
        assert_eq!(Objective::Mnrl.as_str(), "mnrl");
    }

    #[test]
    fn objective_from_str_rejects_an_unknown_value() {
        let err = "cosent".parse::<Objective>().unwrap_err();
        assert!(err.contains("cosent"), "{err}");
        assert!(err.contains("triplet") && err.contains("mnrl"), "{err}");
    }

    /// Projection correctness (unit 63 H4a-delta, task item 4): the MNRL
    /// loader's (anchor, positive) rows must be EXACTLY the same rows, in
    /// the SAME committed order, as the source triplets — negative column
    /// dropped, nothing reordered, nothing dropped or duplicated — against
    /// a small fixture with known ids.
    #[test]
    fn project_to_pairs_keeps_committed_order_and_drops_the_negative_column() {
        let pairs = vec![
            IdTriplet {
                id: "row-0".to_string(),
                anchor: "anchor 0".to_string(),
                positive: "positive 0".to_string(),
                negative: "negative 0".to_string(),
            },
            IdTriplet {
                id: "row-1".to_string(),
                anchor: "anchor 1".to_string(),
                positive: "positive 1".to_string(),
                negative: "negative 1".to_string(),
            },
            IdTriplet {
                id: "row-2".to_string(),
                anchor: "anchor 2".to_string(),
                positive: "positive 2".to_string(),
                negative: "negative 2".to_string(),
            },
        ];
        let projected = project_to_pairs(&pairs);
        assert_eq!(
            projected,
            vec![
                ("anchor 0".to_string(), "positive 0".to_string()),
                ("anchor 1".to_string(), "positive 1".to_string()),
                ("anchor 2".to_string(), "positive 2".to_string()),
            ],
            "projection must be the same rows, same order, negative column dropped"
        );
    }

    /// Negative control on the projection: an empty input projects to an
    /// empty output (no fabricated rows), and the projection never reads
    /// `negative` at all — swapping every `negative` field to a distinct,
    /// unique sentinel value must not change the projected output, proving
    /// the negative column truly plays no role.
    #[test]
    fn project_to_pairs_output_is_independent_of_the_negative_column() {
        let mut pairs = vec![IdTriplet {
            id: "row-0".to_string(),
            anchor: "anchor 0".to_string(),
            positive: "positive 0".to_string(),
            negative: "negative 0".to_string(),
        }];
        let before = project_to_pairs(&pairs);
        pairs[0].negative = "a completely different sentinel negative".to_string();
        let after = project_to_pairs(&pairs);
        assert_eq!(before, after);
        assert!(project_to_pairs(&[]).is_empty());
    }

    /// Unit 63 round-7 audit, finding 1: `--mutant-id`/`--mutant-base-sha`/
    /// `--mutant-patch-sha256` are all-or-none. This check fires FIRST, before
    /// any device/catalog/filesystem setup (see `run_impl`'s own leading
    /// validation block), so a plain `#[test]` (no tokio runtime) suffices —
    /// the function returns before ever reaching a `Handle::current()` call.
    /// Exercises all three "exactly one of three" partial subsets, plus both
    /// "exactly two of three" subsets — every INCOMPLETE non-empty subset a
    /// caller could supply — proving the refusal is not merely reachable for
    /// one particular partial combination.
    #[test]
    fn mutant_provenance_flags_are_refused_when_partially_supplied() {
        let some = |s: &str| Some(s.to_string());
        let combinations: [(Option<String>, Option<String>, Option<String>); 6] = [
            (some("eps-0.10"), None, None),
            (None, some("f".repeat(40).as_str()), None),
            (None, None, some("a".repeat(64).as_str())),
            (some("eps-0.10"), some("f".repeat(40).as_str()), None),
            (some("eps-0.10"), None, some("a".repeat(64).as_str())),
            (
                None,
                some("f".repeat(40).as_str()),
                some("a".repeat(64).as_str()),
            ),
        ];
        for (mutant_id, mutant_base_sha, mutant_patch_sha256) in combinations {
            let mut params = non_perturbation_test_params(std::env::temp_dir());
            params.mutant_id = mutant_id.clone();
            params.mutant_base_sha = mutant_base_sha.clone();
            params.mutant_patch_sha256 = mutant_patch_sha256.clone();
            let result = run_impl(&params, true);
            let err = match result {
                Ok(_) => panic!(
                    "a partial mutant subset must be refused: mutant_id={mutant_id:?}, \
                     mutant_base_sha={mutant_base_sha:?}, \
                     mutant_patch_sha256={mutant_patch_sha256:?}"
                ),
                Err(e) => e,
            };
            let msg = err.to_string();
            assert!(
                msg.contains("all-or-none"),
                "refusal message must name the all-or-none rule: {msg}"
            );
        }
    }

    /// `run_impl`'s `Ok` payload carries a `VarMap`, which has no `Debug`
    /// impl, so the stdlib `Result::expect_err` (which formats the `Ok`
    /// value on failure) cannot be used against it directly — this is the
    /// same shape as the `match result { Ok(_) => panic!(...), Err(e) => e
    /// }` pattern `mutant_provenance_flags_are_refused_when_partially_supplied`
    /// already uses above, pulled out so the round-8 tests below don't each
    /// repeat it.
    fn expect_refused(
        result: Result<(FinetuneRunTier, VarMap), Box<dyn std::error::Error + Send + Sync>>,
        context: &str,
    ) -> Box<dyn std::error::Error + Send + Sync> {
        match result {
            Ok(_) => panic!("{context}"),
            Err(e) => e,
        }
    }

    /// Round-8 audit, finding 4: the all-or-none gate counts PRESENCE
    /// (`is_some()`), not NON-EMPTINESS, so `--mutant-base-sha ""` (which
    /// the CLI happily parses into `Some(String::new())`) sailed through as
    /// "present" and produced a half-labeled leg. All three explicitly
    /// supplied as the empty string must be refused — NOT silently treated
    /// as the ordinary non-mutant (`None`-trio) case, which is a distinct
    /// state (see [`mutant_provenance_all_absent_is_not_the_same_state_as_an_empty_trio`]).
    #[test]
    fn mutant_provenance_empty_string_trio_is_refused() {
        let mut params = non_perturbation_test_params(std::env::temp_dir());
        params.mutant_id = Some(String::new());
        params.mutant_base_sha = Some(String::new());
        params.mutant_patch_sha256 = Some(String::new());
        let err = expect_refused(
            run_impl(&params, true),
            "an explicitly-empty trio must be refused, not treated as absent",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("all-or-none"),
            "refusal message must name the all-or-none rule: {msg}"
        );
        assert!(
            msg.contains("empty-or-whitespace-only"),
            "refusal message must name the empty-or-whitespace-only state: {msg}"
        );
    }

    /// Round-8 audit, finding 4: a whitespace-only trio is exactly as
    /// un-attributable as an empty-string one — `trim()` must run BEFORE the
    /// emptiness check, not after (or not at all), so a padded value can't
    /// slip past as "present".
    #[test]
    fn mutant_provenance_whitespace_trio_is_refused() {
        let mut params = non_perturbation_test_params(std::env::temp_dir());
        params.mutant_id = Some("   ".to_string());
        params.mutant_base_sha = Some("\t\n".to_string());
        params.mutant_patch_sha256 = Some(" ".to_string());
        let err = expect_refused(
            run_impl(&params, true),
            "an explicitly-whitespace trio must be refused, not treated as absent",
        );
        assert!(
            err.to_string().contains("all-or-none"),
            "refusal message must name the all-or-none rule: {}",
            err
        );
    }

    /// Round-8 audit, finding 4: exactly one of the three empty/whitespace
    /// (with the other two genuinely valid) must be refused just like the
    /// classic `None`-mixed-with-`Some` partial subsets above — this is the
    /// specific "half-labeled leg" shape the auditor found the old
    /// `is_some()` count let through (all three `is_some()`, so the OLD
    /// check saw "3 present" and did not refuse it). Exercised in all three
    /// flag positions.
    #[test]
    fn mutant_provenance_one_empty_among_three_is_refused() {
        let valid_id = || Some("eps-0.10".to_string());
        let valid_base = || Some("f".repeat(40));
        let valid_patch = || Some("a".repeat(64));
        let combinations: [(Option<String>, Option<String>, Option<String>); 3] = [
            (Some("  ".to_string()), valid_base(), valid_patch()),
            (valid_id(), Some(String::new()), valid_patch()),
            (valid_id(), valid_base(), Some("\t".to_string())),
        ];
        for (mutant_id, mutant_base_sha, mutant_patch_sha256) in combinations {
            let mut params = non_perturbation_test_params(std::env::temp_dir());
            params.mutant_id = mutant_id.clone();
            params.mutant_base_sha = mutant_base_sha.clone();
            params.mutant_patch_sha256 = mutant_patch_sha256.clone();
            let err = expect_refused(
                run_impl(&params, true),
                &format!(
                    "one empty/whitespace among three (otherwise valid) values must be \
                     refused: mutant_id={mutant_id:?}, mutant_base_sha={mutant_base_sha:?}, \
                     mutant_patch_sha256={mutant_patch_sha256:?}"
                ),
            );
            assert!(
                err.to_string().contains("all-or-none"),
                "refusal message must name the all-or-none rule: {err}"
            );
        }
    }

    /// Round-8 audit, finding 4: malformed (non-hex or wrong-length) shas in
    /// an otherwise-complete, non-empty trio get their own typed, flag-named
    /// refusal — a "3 present" trio is necessary but not sufficient; the
    /// content must actually be a plausible sha. Fires before any
    /// device/catalog/filesystem setup, same as the emptiness gate, so a
    /// plain `#[test]` suffices.
    #[test]
    fn mutant_provenance_malformed_shas_are_refused() {
        let mut too_short_base = non_perturbation_test_params(std::env::temp_dir());
        too_short_base.mutant_id = Some("eps-0.10".to_string());
        too_short_base.mutant_base_sha = Some("abc123".to_string()); // 6 hex chars, below the 7 floor
        too_short_base.mutant_patch_sha256 = Some("a".repeat(64));
        let err = expect_refused(
            run_impl(&too_short_base, true),
            "a too-short mutant-base-sha must be refused",
        );
        assert!(
            err.to_string().contains("--mutant-base-sha"),
            "refusal must name the offending flag: {err}"
        );

        let mut non_hex_base = non_perturbation_test_params(std::env::temp_dir());
        non_hex_base.mutant_id = Some("eps-0.10".to_string());
        non_hex_base.mutant_base_sha = Some("not-a-hex-sha!!".to_string());
        non_hex_base.mutant_patch_sha256 = Some("a".repeat(64));
        let err = expect_refused(
            run_impl(&non_hex_base, true),
            "a non-hex mutant-base-sha must be refused",
        );
        assert!(
            err.to_string().contains("--mutant-base-sha"),
            "refusal must name the offending flag: {err}"
        );

        let mut wrong_len_patch = non_perturbation_test_params(std::env::temp_dir());
        wrong_len_patch.mutant_id = Some("eps-0.10".to_string());
        wrong_len_patch.mutant_base_sha = Some("f".repeat(40));
        wrong_len_patch.mutant_patch_sha256 = Some("a".repeat(63)); // one short of 64
        let err = expect_refused(
            run_impl(&wrong_len_patch, true),
            "a wrong-length mutant-patch-sha256 must be refused",
        );
        assert!(
            err.to_string().contains("--mutant-patch-sha256"),
            "refusal must name the offending flag: {err}"
        );

        let mut non_hex_patch = non_perturbation_test_params(std::env::temp_dir());
        non_hex_patch.mutant_id = Some("eps-0.10".to_string());
        non_hex_patch.mutant_base_sha = Some("f".repeat(40));
        non_hex_patch.mutant_patch_sha256 = Some("z".repeat(64)); // right length, not hex
        let err = expect_refused(
            run_impl(&non_hex_patch, true),
            "a non-hex mutant-patch-sha256 must be refused",
        );
        assert!(
            err.to_string().contains("--mutant-patch-sha256"),
            "refusal must name the offending flag: {err}"
        );
    }

    /// Round-8 audit, finding 4: a fully-supplied, non-empty trio clears the
    /// gate and the STAMPED values (in the returned tier) are the TRIMMED
    /// strings, never the raw, whitespace-padded CLI input — driven through
    /// the REAL `run_impl` end to end (not merely the leading validation
    /// block) over the tiny, CPU-hermetic `tiny_bert` fixture, so this also
    /// re-covers the "all three genuinely present" arm of the gate itself.
    #[tokio::test]
    async fn mutant_provenance_valid_trio_is_stamped_trimmed() {
        let work_dir = tempfile::tempdir().expect("tempdir");
        let mut params = non_perturbation_test_params(work_dir.path().to_path_buf());
        params.mutant_id = Some("  eps-0.10  ".to_string());
        params.mutant_base_sha = Some(format!("  {}  ", "f".repeat(40)));
        params.mutant_patch_sha256 = Some(format!("\t{}\n", "a".repeat(64)));

        let (tier, _varmap) = tokio::task::spawn_blocking(move || run_impl(&params, true))
            .await
            .expect("join run_impl task")
            .expect("a fully-supplied, non-empty (once trimmed) trio must be accepted");

        assert_eq!(tier.mutant_id, Some("eps-0.10".to_string()));
        assert_eq!(tier.mutant_base_sha, Some("f".repeat(40)));
        assert_eq!(tier.mutant_patch_sha256, Some("a".repeat(64)));
    }

    /// Advisory 6 (round-8): this module previously carried
    /// `mutant_provenance_all_absent_clears_the_all_or_none_gate`, which
    /// re-implemented the gate's own `is_some()`-counting predicate INLINE
    /// and asserted it against itself — tautological, since it never called
    /// `run_impl` and so could not observe whether the gate the production
    /// code actually runs lets the all-absent case through. It has been
    /// deleted rather than "fixed", because a real positive control for
    /// "all-absent clears the gate" already exists and is exercised on
    /// every test run:
    ///
    /// - In-process: `init_probe_does_not_perturb_the_training_trajectory_bitwise`
    ///   (this module) drives the REAL `run_impl` end to end, twice, via
    ///   `non_perturbation_test_params` — whose `mutant_id`/`mutant_base_sha`/
    ///   `mutant_patch_sha256` are all `None` — and `.expect()`s `Ok(_)` both
    ///   times. If the all-or-none gate ever wrongly fired on the all-absent
    ///   case, that test would fail immediately with a panic, not silently
    ///   pass.
    /// - End to end via the compiled CLI: every case in
    ///   `finetune_run_smoke.rs` (`finetune_run_smoke_end_to_end_cpu_hermetic`,
    ///   `finetune_run_smoke_mnrl_end_to_end_cpu_hermetic`) invokes the real
    ///   `finetune-run` binary WITHOUT any `--mutant-*` flag and asserts a
    ///   zero exit — the same "never touched" state this module's gate must
    ///   pass.
    #[test]
    fn mutant_provenance_all_absent_is_not_the_same_state_as_an_empty_trio() {
        // Deliberately not a re-implementation of the gate: this pins the
        // TYPE-level distinction the gate itself relies on (never-supplied
        // vs. explicitly-supplied-but-empty) without repeating the gate's
        // own logic. The genuine behavioral proof lives in the tests named
        // in this test's doc comment above.
        let never_touched: Option<String> = None;
        let explicitly_empty: Option<String> = Some(String::new());
        assert_ne!(
            never_touched, explicitly_empty,
            "a caller who never touches a `--mutant-*` flag and a caller who supplies it as an \
             empty string are observably different `Option<String>` values, even though both \
             trim to no content"
        );
    }
}
