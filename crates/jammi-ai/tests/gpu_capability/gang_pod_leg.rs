//! The pod-leg artifact producer for `fine_tune::collective::nccl` (plan
//! #500, U4b's pod-leg acceptance, contract §2c; plan row B8).
//!
//! `ci/scripts/runpod_gpu_gang.sh` (1 pod × 2 A100s, `cargo test … --test
//! gpu_capability gang_ -- --nocapture --test-threads=1`) has run this
//! tree's `gang_` tests on real hardware and passed every one of them, then
//! refused the leg because nothing wrote `$JAMMI_GANG_ARTIFACT_DIR`'s
//! required `gang`-kind artifact (`gang.leg == "pod"`,
//! `ci/scripts/check_cuda_run_artifacts.py`'s rule (k)). [`gang_nccl.rs`]
//! proves the collective itself (rank-ordered sum, unequal-count gather,
//! lockstep flags, abort) over two real devices; it is not this leg's
//! evidence producer — U4b's contract (§2) left the pod leg UNCOVERED. This
//! module is that producer.
//!
//! [`gang_nccl.rs`]: super::gang_nccl
//!
//! # The property (UNITS.md § U4b, the pod-leg acceptance)
//!
//! On a real 2-GPU pod, [`gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1`]
//! trains a real two-rank gang (U4a's single-process multi-GPU
//! `Nccl::single_process`, U4b's `RankContext` + `PartitionSpec::for_gang`)
//! through the production `TrainingLoop::run`, over a real `tiny_bert`
//! projection-head fixture at `lora_dropout = 0.0`, and:
//!
//! - **(a)** runs the SAME two-rank gang TWICE, from the SAME seed: the
//!   published `adapter.safetensors` rank 0 holds after the last step must
//!   be BYTE-IDENTICAL across the two runs (no unseeded RNG on the
//!   trainable-parameter path — family J).
//! - **(c)** compares that gang against a W=1 reference trained at DOUBLE
//!   the per-rank batch (the same global batch, one rank instead of two),
//!   same seed, same data: the per-epoch training loss must agree within
//!   [`GANG_POD_LEG_EPSILON`], the ε [`gather_exactness_w2_matches_
//!   w1_within_pre_registered_epsilon`] (`crates/jammi-ai/src/fine_tune/
//!   trainer.rs`) already pins for the CPU-hermetic form of this same
//!   comparison.
//!
//! The run then writes ONE `gang`-kind, `leg == "pod"` artifact into
//! `$JAMMI_GANG_ARTIFACT_DIR` — [`build_gang_pod_artifact`] /
//! [`write_gang_pod_artifact`] — satisfying every field
//! `ci/scripts/check_cuda_run_artifacts.py`'s rule (k) requires for that
//! leg. A FAILING run (an unequal digest pair, or a delta past ε) writes
//! its own artifact too, `gang.verdict == "fail"` with a `gang.reason`
//! naming what failed and a top-level `status` that is not `GREEN` — never
//! suppressed, because a non-reproducible run's own numbers are exactly the
//! evidence that has to survive.
//!
//! # Deviations from a byte-for-byte U4b/CPU-oracle mirror (with the code
//! cited for each)
//!
//! - **Per-EPOCH, not per-step, loss deltas.** The CPU-hermetic
//!   `gather_exactness_w2_matches_w1_within_pre_registered_epsilon` reads
//!   PER-STEP loss off `TrainingLoop::compute_loss_gathered`/`encode_chunk`
//!   — both private to `trainer.rs` (no `pub` accessor), and this unit's
//!   own scope is `crates/jammi-ai/tests/gpu_capability/**` only (three
//!   concurrent implementers own `worker.rs`/`gang.rs`/`ci/scripts/
//!   runpod_*`; `trainer.rs` is the lead/shared-declaration class for this
//!   wave). The only per-run-progress granularity this test target CAN
//!   observe is the per-EPOCH `avg_train_loss` `harness::loss_capture`
//!   already captures off the trainer's own `tracing::info!("Epoch
//!   complete", …)` event (the SAME mechanism `fine_tune_learns.rs`'s P2
//!   uses). `gang.per_step_loss_delta` therefore carries one entry per
//!   EPOCH here, not per optimizer step — the registry field itself is
//!   untyped prose (`ci/scripts/check_cuda_run_artifacts.py` only requires
//!   a non-empty list of finite numbers), so this is a granularity
//!   deviation from the brief's wording, not a schema violation.
//! - **No `Collective` import.** [`gang_nccl.rs`] imports `fine_tune::
//!   collective::Collective` because it calls collective methods
//!   (`all_reduce_sum`, `all_gather`, …) DIRECTLY on a `Nccl` value. This
//!   module never does: training is driven entirely through
//!   `RankContext`/`TrainingLoop::run`, which invoke the collective
//!   internally: the ONE place this module constructs an `Arc<dyn
//!   Collective>` is `Arc::new(nccl_rank)` handed to `RankContext::new`,
//!   whose parameter type already names the trait, so the unsized
//!   coercion needs no local `use`.
//! - **Own copies of small helpers**, never edits to files this unit does
//!   not own: [`serial_cuda_device_or_require`]/[`second_cuda_device_or_require`]
//!   mirror [`gang_nccl.rs`]'s same-named (but private-to-that-module)
//!   functions; [`claimed_job`]/[`ephemeral_artifact_store`]/
//!   [`ephemeral_hub_source`] mirror `trainer.rs`'s `test_fixtures::
//!   claimed_job` and `gguf_quantized_gpu.rs`'s own private helpers of the
//!   same shape. Each is a small (<20 line), already-proven pattern copied
//!   because the original is private to a file this unit's scope excludes
//!   from editing.
//!
//! # Uncovered
//!
//! - **World ≥ 3.** This leg proves world 2 only (one pod, `gpuCount: 2`) —
//!   the same "known-unmeasured" boundary [`gang_nccl.rs`]'s own module doc
//!   states, filed in the contract of record rather than assumed.
//! - **The cluster (two-host) leg.** Out of scope here; it is U7b-A2b's own
//!   unit and carries a different registry
//!   (`GANG_CLUSTER_FIELD_REGISTRY`, no `digests`/`per_step_loss_delta`/
//!   `epsilon` row at all).
//! - **No CUDA toolchain in this authoring environment.** Every line that
//!   touches a CUDA device, NCCL, or a real `tiny_bert` GPU forward pass is
//!   gated behind `#[cfg(feature = "cuda")]` and is UNEXERCISED here (no
//!   `nvcc`/`libnccl` in this authoring sandbox). What IS exercised here,
//!   on CPU: every non-`cuda` arm type-checks
//!   (`cargo check -p jammi-ai --features live-gpu-tests --test
//!   gpu_capability`), and the artifact's SHAPE is proven by the hermetic
//!   [`synthetic_artifact_tests`] below, which calls the SAME
//!   [`build_gang_pod_artifact`]/[`write_gang_pod_artifact`] the real pod
//!   run calls, from fixed inputs, and validates the result against
//!   `ci/scripts/check_cuda_run_artifacts.py`'s REAL `validate_artifact`
//!   (never a hand port of its rules). The pod run's own numerics
//!   (digest equality, the ε comparison) are therefore UNCOVERED here; the
//!   lead runs the leg on the pod and returns the compiler's/driver's
//!   output on a failure.
//!
//! ## Derivation of [`GANG_POD_LEG_EPSILON`]
//!
//! [`GANG_POD_LEG_EPSILON`] is **2e-4**: double the CPU-hermetic floor
//! already established and tested in this crate
//! (`gather_exactness_w2_matches_w1_within_pre_registered_epsilon`'s
//! `GATHER_EXACTNESS_EPSILON = 1e-4`, `crates/jammi-ai/src/fine_tune/
//! trainer.rs` — itself `batch_bucket.rs`'s own bucket-padding-variance
//! `TOLERANCE`), to admit exactly the ONE further fp32 reduction
//! reassociation a real GPU pod run adds beyond what that CPU oracle
//! already measures: cuBLAS's own forward/backward accumulation order, plus
//! NCCL's ring/tree `ncclAllReduce` summation order, neither of which is
//! bound to match candle's CPU rank-ordered fold the hermetic oracle
//! exercises.
//!
//! That additional term is bounded analytically, not merely asserted: a
//! fp32 reassociation error over `n` summed terms is bounded by
//! `(n - 1) * eps_f32 * max|term|`, `eps_f32 = 2^-23 ≈ 1.19e-7`. This
//! fixture's global batch (world 2 × per-rank batch 2) sums at most 4
//! terms, so the bound is `≈ 3 * 1.19e-7 * O(1) ≈ 3.6e-7` — three orders of
//! magnitude under the 2e-4 registered here, leaving ample headroom while
//! staying more than three orders of magnitude BELOW the divergence a real
//! bug produces: `gather_exactness_w2_matches_w1_within_pre_registered_
//! epsilon`'s own EXECUTED red-proof (the gather replaced with a `.clone()`
//! that never runs it) measured a gathered-vs-reference loss divergence of
//! `0.37497652` vs `1.0657526` — an O(1) mistake, not a reassociation-noise
//! one. 2e-4 is therefore discriminative: it tolerates the extra GPU fold
//! this leg adds while remaining far below any gradient-routing hazard.
//!
//! This ε was registered in its own commit (`6a9ced97614890d8c50880b33b3bd70d2e2a67d1`,
//! see [`GANG_POD_LEG_EPSILON_REGISTERED_SHA`]), landed BEFORE this test and
//! this ε's own use in a measured run existed — the pre-registration
//! `ci/scripts/check_cuda_run_artifacts.py`'s rule (k) requires.

use crate::skip_without_gpu;

#[cfg(feature = "cuda")]
use jammi_ai::fine_tune::collective::nccl::Nccl;
#[cfg(feature = "cuda")]
use jammi_ai::fine_tune::collective::BlockingCall;
#[cfg(feature = "cuda")]
use jammi_ai::fine_tune::partition::{PartitionRule, PartitionSpec};
#[cfg(feature = "cuda")]
use jammi_ai::fine_tune::trainer::{RankContext, TrainingLoopBuilder, TrainingResult};

/// The pre-registered ε for the pod-leg's W=2×B vs W=1×2B per-epoch
/// training-loss reproducibility bound. See the module doc's "Derivation".
pub(crate) const GANG_POD_LEG_EPSILON: f32 = 2.0e-4;

/// [`GANG_POD_LEG_EPSILON`]'s own derivation, restated as the exact
/// `gang.epsilon.derivation` string the committed artifact carries — kept
/// as one constant so the module doc above and the artifact's own field
/// can never drift apart from each other.
pub(crate) const GANG_POD_LEG_EPSILON_DERIVATION: &str = "2e-4 = 2x the CPU-hermetic gather-exactness floor (1e-4, gather_exactness_w2_matches_w1_within_pre_registered_epsilon's GATHER_EXACTNESS_EPSILON in crates/jammi-ai/src/fine_tune/trainer.rs), admitting exactly one further fp32 reduction reassociation a real GPU pod run adds beyond that CPU oracle: cuBLAS's own forward/backward accumulation order plus NCCL's ring/tree ncclAllReduce summation order. Analytically bounded: an (n-1)*eps_f32*max|term| reassociation-error bound (eps_f32 = 2^-23 ~= 1.19e-7, n <= 4 terms this fixture's global batch sums) is ~= 3.6e-7, three orders of magnitude under 2e-4, while staying more than three orders of magnitude below the O(1) divergence a real gradient-routing bug produces (the same oracle's own executed red-proof measured 0.37497652 vs 1.0657526).";

/// The commit [`GANG_POD_LEG_EPSILON`] was registered at — this unit's
/// FIRST commit, landed before this test (and the pod run it gates) ever
/// existed. `ci/scripts/check_cuda_run_artifacts.py`'s rule (k)
/// (`_gang_check_epsilon`/`_gang_evidence_anchor`) requires this to be a
/// STRICT ancestor of the artifact's own measured `git_sha`.
pub(crate) const GANG_POD_LEG_EPSILON_REGISTERED_SHA: &str =
    "e7440afd5398ae08d531df1826f92efc946fe2ac";

/// This leg's sole registered producer path —
/// `GANG_LEG_PRODUCER_PATH[GANG_LEG_POD]` in
/// `ci/scripts/check_cuda_run_artifacts.py` (F4: a self-declared `gang.leg`
/// cannot point at a different, or no, driver).
const GANG_POD_LEG_PRODUCER_PATH: &str = "ci/scripts/runpod_gpu_gang.sh";

/// `tiny_bert`'s hidden width — the same fixture and constant
/// `trainer.rs`'s `gang_determinism_oracle::HIDDEN` uses.
#[cfg(feature = "cuda")]
const TINY_BERT_HIDDEN: usize = 32;

/// Per-rank micro-batch (`B`) for the W=2 gang; the W=1 reference trains at
/// double this (the SAME global batch, one rank instead of two).
const POD_LEG_PER_RANK_BATCH: usize = 2;

/// Training rows — divisible by both the W=2 gang's global batch
/// (`2 * POD_LEG_PER_RANK_BATCH` = 4) and the W=1 reference's batch (4), so
/// every step of both runs holds an EQUAL row count (no remainder-batch
/// confound in this comparison).
#[cfg(feature = "cuda")]
const POD_LEG_TRAIN_ROWS: usize = 8;

/// Epochs — small (pod time is billed), but ≥ 2 so the per-epoch loss
/// delta series carries more than one comparison point.
#[cfg(feature = "cuda")]
const POD_LEG_EPOCHS: usize = 2;

/// The seed both same-seed W=2 runs (the digest pair) and the W=1
/// reference share. Any fixed value works — the property is EQUALITY
/// across runs at this seed, never this value's own magnitude.
#[cfg(feature = "cuda")]
const POD_LEG_SEED: u64 = 4242;

// ─── The artifact writer (pure — no I/O, no CUDA; the hermetic oracle below
// exercises this SAME code the real pod run calls) ─────────────────────────

/// One rank's device entry in `gang.ranks` —
/// `ci/scripts/check_cuda_run_artifacts.py`'s `_gang_check_ranks`.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodRank {
    rank: u32,
    device: String,
}

/// One entry of the same-seed digest PAIR — `_gang_check_digests`.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodDigest {
    seed: u64,
    digest: String,
}

/// The pre-registered tolerance — `_gang_check_epsilon`.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodEpsilon {
    value: f32,
    derivation: String,
    registered_sha: String,
}

/// The `gang` block itself — `GANG_POD_FIELD_REGISTRY`.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodBlock {
    leg: &'static str,
    world: u32,
    collective: &'static str,
    ranks: Vec<GangPodRank>,
    digests: Vec<GangPodDigest>,
    per_step_loss_delta: Vec<f64>,
    epsilon: GangPodEpsilon,
    verdict: &'static str,
    reason: String,
}

/// This leg's sole registered producer.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodProducer {
    path: &'static str,
    kind: &'static str,
    invocation: &'static str,
    gating: &'static str,
}

/// The full committed artifact document — every top-level field rule (a)
/// requires, plus the `gang` block rule (k) requires for `leg == "pod"`.
#[derive(Debug, Clone, serde::Serialize)]
struct GangPodArtifact {
    schema_version: u32,
    git_sha: String,
    #[serde(rename = "box")]
    box_name: String,
    producer: GangPodProducer,
    status: &'static str,
    artifact_kind: &'static str,
    gang: GangPodBlock,
}

/// [`build_gang_pod_artifact`]'s plain-data inputs — everything the writer
/// needs, with no I/O of its own, so the SAME function assembles both the
/// real pod run's artifact and the hermetic synthetic one below.
struct GangPodArtifactInputs {
    git_sha: String,
    box_name: String,
    ranks: Vec<(u32, String)>,
    digests: [(u64, String); 2],
    per_step_loss_delta: Vec<f64>,
    verdict_pass: bool,
    reason: String,
}

/// Assemble the pod-leg `gang` artifact document from plain data — pure, no
/// I/O, so it is exercised hermetically on CPU
/// ([`synthetic_artifact_tests`]) with the SAME code the real pod run
/// calls.
fn build_gang_pod_artifact(inputs: GangPodArtifactInputs) -> GangPodArtifact {
    GangPodArtifact {
        schema_version: 1,
        git_sha: inputs.git_sha,
        box_name: inputs.box_name,
        producer: GangPodProducer {
            path: GANG_POD_LEG_PRODUCER_PATH,
            kind: "script",
            invocation: "bash ci/scripts/runpod_gpu_gang.sh",
            gating: "none",
        },
        status: if inputs.verdict_pass { "GREEN" } else { "RED" },
        artifact_kind: "gang",
        gang: GangPodBlock {
            leg: "pod",
            world: 2,
            collective: "nccl",
            ranks: inputs
                .ranks
                .into_iter()
                .map(|(rank, device)| GangPodRank { rank, device })
                .collect(),
            digests: inputs
                .digests
                .into_iter()
                .map(|(seed, digest)| GangPodDigest { seed, digest })
                .collect(),
            per_step_loss_delta: inputs.per_step_loss_delta,
            epsilon: GangPodEpsilon {
                value: GANG_POD_LEG_EPSILON,
                derivation: GANG_POD_LEG_EPSILON_DERIVATION.to_string(),
                registered_sha: GANG_POD_LEG_EPSILON_REGISTERED_SHA.to_string(),
            },
            verdict: if inputs.verdict_pass { "pass" } else { "fail" },
            reason: inputs.reason,
        },
    }
}

/// The committed filename `GANG_ARTIFACT_FILENAME_RE` matches:
/// `<date>-500-u4b-gang-pod-<sha8>-a100-sxm4.json`.
fn gang_pod_artifact_filename(git_sha: &str) -> String {
    let date = chrono::Utc::now().format("%Y-%m-%d");
    let sha8 = &git_sha[..git_sha.len().min(8)];
    format!("{date}-500-u4b-gang-pod-{sha8}-a100-sxm4.json")
}

/// Write `artifact` into `dir` (created if missing), returning the path
/// written — the ONE artifact `ci/scripts/runpod_gpu_gang.sh` requires to
/// exist under `$JAMMI_GANG_ARTIFACT_DIR` before it will call the leg
/// proven.
fn write_gang_pod_artifact(
    dir: &std::path::Path,
    artifact: &GangPodArtifact,
) -> std::io::Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(gang_pod_artifact_filename(&artifact.git_sha));
    let body = serde_json::to_vec_pretty(artifact)
        .unwrap_or_else(|e| panic!("GangPodArtifact must always serialize: {e}"));
    std::fs::write(&path, body)?;
    Ok(path)
}

// ─── General utilities (no CUDA; used by both the real test and the
// hermetic oracle) ──────────────────────────────────────────────────────────

/// Workspace root — three levels up from this test's manifest dir, matching
/// `harness::workspace_root`'s own (private-to-`harness`) computation; kept
/// as a second copy here because this module needs it for `git`
/// operations `harness` has no reason to expose.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// `git rev-parse HEAD` in the workspace root — the measured tree's own
/// sha, read live rather than baked in, exactly as
/// `ci/scripts/runpod_gpu_gang.sh` prints its own
/// `PROVE_SHA=$(git rev-parse HEAD)`.
fn git_head_sha() -> String {
    let root = workspace_root();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run `git rev-parse HEAD` in {}: {e}",
                root.display()
            )
        });
    assert!(
        output.status.success(),
        "`git rev-parse HEAD` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap_or_else(|e| panic!("`git rev-parse HEAD` printed non-UTF-8: {e}"))
        .trim()
        .to_string()
}

// ─── CUDA-only machinery: device acquisition, model loading, the per-rank
// driver, and the two run shapes (W=2 gang, W=1 reference) ─────────────────

/// [`crate::harness::serial_cuda_device`], hard-failing under
/// `JAMMI_REQUIRE_CUDA` rather than skipping — the SAME require-gate idiom
/// `gang_nccl.rs::serial_cuda_device_or_require` uses (that function is
/// private to `gang_nccl.rs`, out of this unit's file scope, so this is a
/// deliberate second copy of an already-proven ~10-line pattern, not a
/// re-derivation). `ci/scripts/runpod_gpu_gang.sh` exports
/// `JAMMI_REQUIRE_CUDA=1`, so the pod's own run hard-fails rather than
/// skipping.
#[cfg(feature = "cuda")]
fn serial_cuda_device_or_require(test: &str) -> Option<crate::harness::SerialGpu> {
    match crate::harness::serial_cuda_device() {
        Some(slot) => Some(slot),
        None => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_CUDA is set but no usable CUDA device could be \
                     acquired — a silent skip is not acceptable here"
                );
            }
            None
        }
    }
}

/// A second CUDA device, hard-failing under `JAMMI_REQUIRE_CUDA_GANG` —
/// mirrors `gang_nccl.rs::second_cuda_device_or_require` (private to that
/// module; see [`serial_cuda_device_or_require`]'s own doc for why this is
/// a second copy). `ci/scripts/runpod_gpu_gang.sh` exports
/// `JAMMI_REQUIRE_CUDA_GANG=1`.
#[cfg(feature = "cuda")]
fn second_cuda_device_or_require(test: &str) -> Option<candle_core::Device> {
    match candle_core::Device::new_cuda(1) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA_GANG").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_CUDA_GANG is set but a second CUDA device could not \
                     be acquired — a two-rank NCCL gang needs two visible devices: {e}"
                );
            }
            None
        }
    }
}

/// An in-memory `ArtifactStore` — only [`jammi_ai::model::resolver::
/// ModelResolver::new`] requires one (this leg checkpoints nothing durable);
/// mirrors `gguf_quantized_gpu.rs::ephemeral_artifact_store` (private to
/// that module).
#[cfg(feature = "cuda")]
fn ephemeral_artifact_store() -> std::sync::Arc<jammi_db::store::ArtifactStore> {
    let cache = tempfile::tempdir().unwrap().keep();
    std::sync::Arc::new(
        jammi_db::store::ArtifactStore::with_root(
            jammi_db::storage::StorageUrl::memory("gang-pod-leg-test-artifacts"),
            jammi_db::storage::StorageRegistry::new(),
            cache,
        )
        .unwrap(),
    )
}

/// A `HubSource` rooted at a fresh tempdir, resolving no remote name —
/// mirrors `gguf_quantized_gpu.rs::ephemeral_hub_source` (private to that
/// module); this leg loads only the local `tiny_bert` cookbook fixture.
#[cfg(feature = "cuda")]
fn ephemeral_hub_source() -> jammi_ai::model::hub::HubSource {
    let root = tempfile::tempdir().unwrap().keep();
    jammi_ai::model::hub::HubSource::from_config(
        &jammi_db::config::ModelsConfig {
            hub_cache_dir: Some(root),
            ..Default::default()
        },
        &|_: &str| None,
    )
    .unwrap()
}

/// A catalog holding a registered model and a job `tag` claimed by
/// `{tag}-worker` — mirrors `trainer.rs`'s `test_fixtures::claimed_job`
/// (private to `trainer.rs`, a shared-declaration file this unit does not
/// edit). Returns the catalog and the tempdir backing it.
#[cfg(feature = "cuda")]
async fn claimed_job(
    tag: &str,
) -> (
    std::sync::Arc<jammi_db::catalog::Catalog>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let catalog = std::sync::Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
    let model_id = format!("{tag}-model");
    catalog
        .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &model_id,
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: jammi_ai::model::ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
            job_id: tag,
            kind: "fine_tune",
            execution: jammi_db::catalog::status::JobExecution::Queued,
            spec: "{}",
            model_ref: Some(&format!("{model_id}::1")),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    catalog
        .claim_next(
            &format!("{tag}-worker"),
            &["fine_tune"],
            std::time::Duration::from_secs(60),
        )
        .await
        .unwrap()
        .expect("queued job is claimable");
    (catalog, dir)
}

/// Load the `tiny_bert` cookbook fixture directly onto CUDA device
/// `device_ordinal`, through the same resolve→backend-load path serving
/// uses (`jammi_ai::model::resolver::ModelResolver` +
/// `jammi_ai::model::backend::candle::CandleBackend`) — the
/// `gguf_quantized_gpu.rs` precedent for placing a `LoadedModel` on a
/// NAMED device rather than through the async `ModelCache`'s scheduler
/// budgets (which this leg's two-rank, two-device placement does not need
/// to route through).
#[cfg(feature = "cuda")]
async fn load_tiny_bert_on(device_ordinal: i32) -> std::sync::Arc<jammi_ai::model::LoadedModel> {
    use jammi_ai::model::backend::candle::CandleBackend;
    use jammi_ai::model::backend::{DeviceConfig, ModelBackend};
    use jammi_ai::model::resolver::ModelResolver;
    use jammi_ai::model::{BackendType, ModelSource, ModelTask};

    let catalog_dir = tempfile::tempdir().unwrap();
    let catalog = std::sync::Arc::new(
        jammi_db::catalog::Catalog::open(catalog_dir.path())
            .await
            .unwrap(),
    );
    let resolver = ModelResolver::new(catalog, ephemeral_artifact_store(), ephemeral_hub_source())
        .unwrap_or_else(|e| panic!("ModelResolver::new failed: {e}"));
    let source = ModelSource::local(crate::harness::cookbook_fixture("tiny_bert"));
    let resolved = resolver
        .resolve(&source, ModelTask::TextEmbedding, Some(BackendType::Candle))
        .await
        .unwrap_or_else(|e| panic!("failed to resolve the tiny_bert fixture: {e}"));
    let backend = CandleBackend;
    let device_config = DeviceConfig {
        gpu_device: device_ordinal,
        devices: vec![device_ordinal],
        memory_fraction: 1.0,
        require_gpu: true,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let loaded = backend
        .load(&resolved, &device_config)
        .unwrap_or_else(|e| panic!("failed to load tiny_bert onto cuda:{device_ordinal}: {e}"));
    std::sync::Arc::new(loaded)
}

/// `FineTuneConfig` for the W=2 gang: per-rank batch [`POD_LEG_PER_RANK_BATCH`],
/// `lora_dropout = 0.0` (this leg's acceptance pins it — U4b's per-rank
/// dropout Philox split is `dropout_seed_split_*`'s own property, not
/// this one's).
#[cfg(feature = "cuda")]
fn gang_pod_config() -> jammi_ai::fine_tune::FineTuneConfig {
    jammi_ai::fine_tune::FineTuneConfig {
        epochs: POD_LEG_EPOCHS,
        batch_size: POD_LEG_PER_RANK_BATCH,
        validation_fraction: 0.0,
        warmup_steps: 0,
        gradient_accumulation_steps: 1,
        lora_rank: 2,
        lora_dropout: 0.0,
        seed: POD_LEG_SEED,
        early_stopping_metric: jammi_ai::fine_tune::EarlyStoppingMetric::TrainLoss,
        early_stopping_patience: 10_000,
        learning_rate: 1e-4,
        ..Default::default()
    }
}

/// [`gang_pod_config`] at DOUBLE the per-rank batch — the W=1 reference's
/// own global batch, unchanged otherwise (same seed, same epochs, same
/// data).
#[cfg(feature = "cuda")]
fn gang_pod_reference_config() -> jammi_ai::fine_tune::FineTuneConfig {
    jammi_ai::fine_tune::FineTuneConfig {
        batch_size: POD_LEG_PER_RANK_BATCH * 2,
        ..gang_pod_config()
    }
}

/// [`POD_LEG_TRAIN_ROWS`] anchor/positive pairs — the same
/// `TrainingDataLoader::from_pairs` shape `trainer.rs`'s
/// `gang_determinism_oracle::pairs` uses.
#[cfg(feature = "cuda")]
fn pod_leg_pairs() -> jammi_ai::fine_tune::data::TrainingDataLoader {
    jammi_ai::fine_tune::data::TrainingDataLoader::from_pairs(
        (0..POD_LEG_TRAIN_ROWS)
            .map(|i| {
                (
                    jammi_test_utils::tiny_vocab_text('a', i),
                    jammi_test_utils::tiny_vocab_text('p', i),
                )
            })
            .collect(),
    )
}

/// Run one rank (gang or single-rank reference) of a real fine-tune through
/// the production `TrainingLoop::run`, on its own tokio runtime — mirrors
/// `trainer.rs`'s `gang_determinism_oracle::run_gang_rank`'s own
/// build-inside-`block_on`-then-`rt.enter()`-then-call-`run`-synchronously
/// shape (a second copy: that function is private to `trainer.rs`).
/// Returns the [`TrainingResult`] (its `artifact_dir` holds the published
/// `adapter.safetensors`).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_pod_rank(
    call: BlockingCall,
    tag: String,
    job_id: String,
    config: jammi_ai::fine_tune::FineTuneConfig,
    loader: jammi_ai::fine_tune::data::TrainingDataLoader,
    rank_ctx: RankContext,
    device: candle_core::Device,
    device_ordinal: i32,
) -> TrainingResult {
    use jammi_ai::fine_tune::lora::build_projection_head_for_rank;
    use jammi_ai::fine_tune::source::TrainingSource;
    use jammi_ai::fine_tune::target::TrainingTarget;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("{tag}: failed to build a tokio runtime: {e}"));
    let mut loop_ = rt.block_on(async {
        let base_model = load_tiny_bert_on(device_ordinal).await;
        let (catalog, dir) = claimed_job(&tag).await;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let dropout_seed = rank_ctx.dropout_seed(config.seed);
        let head =
            build_projection_head_for_rank(TINY_BERT_HIDDEN, &config, &varmap, &vb, dropout_seed)
                .unwrap_or_else(|e| panic!("{tag}: build_projection_head_for_rank failed: {e}"));
        TrainingLoopBuilder::new(TrainingTarget::ProjectionHead { head }, varmap, config)
            .device(device.clone())
            .job_id(job_id.clone())
            .worker_id(format!("{tag}-worker"))
            .catalog(catalog)
            .artifact_dir(dir.path().to_path_buf())
            .base_model(base_model)
            .rank_context(rank_ctx)
            .build()
            .unwrap_or_else(|e| panic!("{tag}: TrainingLoopBuilder::build failed: {e}"))
    });
    let _enter = rt.enter();
    loop_
        .run(&call, TrainingSource::Resident(loader))
        .unwrap_or_else(|e| panic!("{tag}: TrainingLoop::run failed: {e}"))
}

/// `(epoch, loss)` rows deduped to ONE entry per epoch — a W=2 gang's two
/// ranks each emit an "Epoch complete" event into the SAME process-global
/// `harness::loss_capture` buffer (they are synchronized by the gather, so
/// both entries for one epoch should agree); this keeps the first-seen
/// value per epoch, in epoch order.
#[cfg(feature = "cuda")]
fn dedup_by_epoch(curve: &[(u64, f64)]) -> Vec<(u64, f64)> {
    let mut by_epoch: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
    for &(epoch, loss) in curve {
        by_epoch.entry(epoch).or_insert(loss);
    }
    by_epoch.into_iter().collect()
}

/// SHA-256 hex digest of `bytes` — used over the published
/// `adapter.safetensors` bytes (the "canonical adapter bytes rank 0 holds
/// after the last step").
#[cfg(feature = "cuda")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// A named `nvidia-smi` field for GPU `ordinal`, or `"unknown"` if the
/// query fails (never masked as a panic — this is descriptive metadata for
/// the artifact's `box`/`gang.ranks[].device`, not a correctness input).
#[cfg(feature = "cuda")]
fn nvidia_smi_field(ordinal: i32, field: &str) -> String {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={field}"),
            "--format=csv,noheader",
            &format!("--id={ordinal}"),
        ])
        .output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

/// A free-text description of the box this leg ran on — `nvidia-smi`'s own
/// device name + driver version, matching the free-text convention every
/// other committed artifact under `crates/jammi-kernels/artifacts/
/// cuda-runs/` uses for its own `box` field.
#[cfg(feature = "cuda")]
fn box_description() -> String {
    format!(
        "{} x2, driver {}",
        nvidia_smi_field(0, "name"),
        nvidia_smi_field(0, "driver_version"),
    )
}

/// Run a real two-rank W=2 gang once (both ranks over a real NCCL
/// communicator, one device each), returning rank 0's published-adapter
/// sha256 digest and the deduped per-epoch loss curve.
#[cfg(feature = "cuda")]
fn run_pod_gang(
    run_tag: &str,
    dev0: &candle_core::Device,
    dev1: &candle_core::Device,
    loader: fn() -> jammi_ai::fine_tune::data::TrainingDataLoader,
) -> (String, Vec<(u64, f64)>) {
    crate::harness::loss_capture::reset();
    let devices = [dev0.clone(), dev1.clone()];
    let ranks = Nccl::single_process(&devices)
        .unwrap_or_else(|e| panic!("{run_tag}: ncclCommInitAll failed: {e}"));
    let job_id = format!("{run_tag}-job");
    let handles: Vec<_> = ranks
        .into_iter()
        .zip(devices.iter().cloned())
        .enumerate()
        .map(|(idx, (nccl_rank, device))| {
            let partition = PartitionSpec::for_gang(
                idx,
                2,
                POD_LEG_PER_RANK_BATCH,
                PartitionRule::BlockByGlobalBatch,
            )
            .unwrap_or_else(|e| panic!("{run_tag}: PartitionSpec::for_gang failed: {e}"));
            let rank_ctx = RankContext::new(std::sync::Arc::new(nccl_rank), partition);
            let tag = format!("{run_tag}-r{idx}");
            let job_id = job_id.clone();
            let config = gang_pod_config();
            let loader = loader();
            let ordinal = idx as i32;
            BlockingCall::spawn_thread(move |call| {
                run_pod_rank(call, tag, job_id, config, loader, rank_ctx, device, ordinal)
            })
        })
        .collect();

    let results: Vec<TrainingResult> = handles
        .into_iter()
        .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
        .collect();

    let adapter_bytes = std::fs::read(results[0].artifact_dir.path().join("adapter.safetensors"))
        .unwrap_or_else(|e| panic!("{run_tag}: failed to read rank 0's adapter.safetensors: {e}"));
    let digest = sha256_hex(&adapter_bytes);
    let curve = dedup_by_epoch(&crate::harness::loss_capture::captured());
    (digest, curve)
}

/// Run the W=1 reference once (single rank, double the per-rank batch, on
/// `device_ordinal`), returning its deduped per-epoch loss curve.
#[cfg(feature = "cuda")]
fn run_pod_reference(
    run_tag: &str,
    device: &candle_core::Device,
    device_ordinal: i32,
    loader: fn() -> jammi_ai::fine_tune::data::TrainingDataLoader,
) -> Vec<(u64, f64)> {
    crate::harness::loss_capture::reset();
    let config = gang_pod_reference_config();
    let loader = loader();
    let job_id = format!("{run_tag}-job");
    let tag = run_tag.to_string();
    let device_clone = device.clone();
    let rank_ctx = RankContext::single_rank(config.batch_size, PartitionRule::BlockByGlobalBatch);
    let handle = BlockingCall::spawn_thread(move |call| {
        run_pod_rank(
            call,
            tag,
            job_id,
            config,
            loader,
            rank_ctx,
            device_clone,
            device_ordinal,
        )
    });
    let _result = handle
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e));
    dedup_by_epoch(&crate::harness::loss_capture::captured())
}

/// The pod-leg producer test: two same-seed W=2 gangs (a reproducible
/// digest pair) plus a W=1×2B reference (the per-epoch loss-delta bound),
/// writing the registry's `gang` artifact into `$JAMMI_GANG_ARTIFACT_DIR`
/// on both the pass and the fail arm. See the module doc for the full
/// property and its deviations/uncovered determinants.
///
/// `JAMMI_GANG_ARTIFACT_DIR` unset: the assertions below still execute in
/// full (this test still proves the property on this run), but nothing is
/// written — stated via a loud `tracing::warn`, never silently skipped.
/// A graph-sampled training set for the pod leg: an 8-node ring sampled by
/// seeded walks into `(anchor, positive)` pairs — the loader a `graph_fine_tune`
/// job trains from. Its rows are a whole number of global batches (see the
/// test below), so the gang and the double-batch reference step in lockstep.
fn pod_leg_graph_sample() -> jammi_ai::fine_tune::data::TrainingDataLoader {
    use jammi_ai::fine_tune::graph_sampler::{
        sort_into_graph_read_order, GraphEdge, GraphSampleConfig, GraphSampler, TextNode,
    };
    const NODES: usize = 8;
    let mut nodes: Vec<TextNode> = (0..NODES)
        .map(|i| TextNode::new(format!("g{i}"), jammi_test_utils::tiny_vocab_text('g', i)))
        .collect();
    let mut edges: Vec<GraphEdge> = (0..NODES)
        .flat_map(|i| {
            let next = (i + 1) % NODES;
            [
                GraphEdge::declared(format!("g{i}"), format!("g{next}")),
                GraphEdge::declared(format!("g{next}"), format!("g{i}")),
            ]
        })
        .collect();
    sort_into_graph_read_order(&mut nodes, &mut edges);
    let config = GraphSampleConfig {
        walk_length: 2,
        walks_per_node: 1,
        hard_negatives: 0,
        exclude_hops: 1,
        min_negatives: 1,
        seed: 7,
        ..GraphSampleConfig::default()
    };
    let sampler = GraphSampler::build(nodes, edges, config).expect("the ring is a valid graph");
    jammi_ai::fine_tune::data::TrainingDataLoader::from_graph(&sampler)
        .expect("the ring samples at least one pair")
}

/// Runs on any host: the graph fixture must divide into whole global batches
/// (`W = 2` ranks x [`POD_LEG_PER_RANK_BATCH`]), or the gang and its W=1
/// reference would not see the same steps — found here, not on a rented pod.
#[test]
fn pod_leg_graph_sample_is_a_whole_number_of_global_batches() {
    let rows = pod_leg_graph_sample().len();
    let global_batch = 2 * POD_LEG_PER_RANK_BATCH;
    assert!(
        rows >= 2 * global_batch,
        "{rows} rows is under two global batches"
    );
    assert_eq!(
        rows % global_batch,
        0,
        "{rows} sampled rows do not divide into global batches of {global_batch}"
    );
}

/// The pod leg's property for a GRAPH-SAMPLED training set: on two real GPUs
/// over NCCL, a two-rank gang reproduces itself byte for byte, and its loss
/// curve matches a single rank at double the batch — the reduced gradient is
/// the single-rank gradient over the union batch. The pairs test below owns
/// this leg's committed artifact; this one asserts the same three properties
/// for the loader a `graph_fine_tune` job trains from.
#[test]
fn gang_pod_leg_graph_sample_two_ranks_reproduce_and_match_w1() {
    skip_without_gpu!();

    #[cfg(feature = "cuda")]
    {
        const TEST: &str = "gang_pod_leg_graph_sample_two_ranks_reproduce_and_match_w1";
        crate::harness::loss_capture::install();

        let Some(slot) = serial_cuda_device_or_require(TEST) else {
            tracing::warn!("SKIP: no usable CUDA device");
            return;
        };
        let dev0 = slot.device().clone();
        let Some(dev1) = second_cuda_device_or_require(TEST) else {
            tracing::warn!("SKIP: a two-rank NCCL gang needs two CUDA devices; this host has one");
            return;
        };

        let (digest_a, curve_a) =
            run_pod_gang("podleg-graph-a", &dev0, &dev1, pod_leg_graph_sample);
        let (digest_b, _) = run_pod_gang("podleg-graph-b", &dev0, &dev1, pod_leg_graph_sample);
        let curve_ref = run_pod_reference("podleg-graph-ref", &dev0, 0, pod_leg_graph_sample);

        assert!(
            !curve_a.is_empty() && !curve_ref.is_empty(),
            "{TEST}: no per-epoch loss was captured (gang {curve_a:?}, reference {curve_ref:?})"
        );
        assert_eq!(
            digest_a, digest_b,
            "{TEST}: two same-seed gangs published different adapters"
        );
        assert_eq!(
            curve_a.len(),
            curve_ref.len(),
            "{TEST}: the gang and the W=1 reference ran a different number of epochs"
        );
        let worst = curve_a
            .iter()
            .zip(&curve_ref)
            .map(|((_, gang), (_, reference))| (gang - reference).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst <= GANG_POD_LEG_EPSILON as f64,
            "{TEST}: the gang's loss curve leaves the W=1 double-batch reference by {worst:e} \
             (allowed {GANG_POD_LEG_EPSILON:e}): gang {curve_a:?}, reference {curve_ref:?}"
        );
    }
}

#[test]
fn gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1() {
    skip_without_gpu!();

    let artifact_dir = std::env::var_os("JAMMI_GANG_ARTIFACT_DIR").map(std::path::PathBuf::from);
    if artifact_dir.is_none() {
        tracing::warn!(
            "JAMMI_GANG_ARTIFACT_DIR is unset — this run's assertions still execute in full, but \
             no artifact will be written (see the module doc)"
        );
    }

    #[cfg(feature = "cuda")]
    {
        const TEST: &str = "gang_pod_leg_two_ranks_over_nccl_reproduce_and_match_w1";
        crate::harness::loss_capture::install();

        let Some(slot) = serial_cuda_device_or_require(TEST) else {
            tracing::warn!("SKIP: no usable CUDA device");
            return;
        };
        let dev0 = slot.device().clone();
        let Some(dev1) = second_cuda_device_or_require(TEST) else {
            tracing::warn!("SKIP: a two-rank NCCL gang needs two CUDA devices; this host has one");
            return;
        };

        // (a) two independent same-seed W=2 gangs -> a reproducible digest pair.
        let (digest_a, curve_a) = run_pod_gang("podleg-a", &dev0, &dev1, pod_leg_pairs);
        let (digest_b, _curve_b) = run_pod_gang("podleg-b", &dev0, &dev1, pod_leg_pairs);

        // (c) the W=1 x 2B reference, same seed, same data.
        let curve_ref = run_pod_reference("podleg-ref", &dev0, 0, pod_leg_pairs);

        assert!(
            !curve_a.is_empty() && !curve_ref.is_empty(),
            "{TEST}: the harness captured no per-epoch loss at all (curve_a={curve_a:?}, \
             curve_ref={curve_ref:?}) — a fixture/harness defect, not a property this leg can \
             call pass or fail"
        );

        let digest_ok = digest_a == digest_b;
        let same_length = curve_a.len() == curve_ref.len();
        let deltas: Vec<f64> = curve_a
            .iter()
            .zip(curve_ref.iter())
            .map(|((_, a), (_, r))| (a - r).abs())
            .collect();
        let worst = deltas.iter().cloned().fold(0.0_f64, f64::max);
        let within_epsilon = !deltas.is_empty() && worst <= GANG_POD_LEG_EPSILON as f64;

        let mut reasons = Vec::new();
        if !digest_ok {
            reasons.push(format!(
                "the same-seed W=2 digest pair disagreed: {digest_a} vs {digest_b}"
            ));
        }
        if !same_length {
            reasons.push(format!(
                "the W=2 curve has {} epoch(s), the W=1 reference has {} — cannot compare per-step",
                curve_a.len(),
                curve_ref.len()
            ));
        }
        if same_length && !within_epsilon {
            reasons.push(format!(
                "worst per-epoch loss delta {worst} exceeds the pre-registered epsilon \
                 {GANG_POD_LEG_EPSILON}"
            ));
        }
        let verdict_pass = digest_ok && same_length && within_epsilon;
        let reason = reasons.join("; ");

        let git_sha = git_head_sha();
        let inputs = GangPodArtifactInputs {
            git_sha: git_sha.clone(),
            box_name: box_description(),
            ranks: vec![
                (0, format!("cuda:0 {}", nvidia_smi_field(0, "name"))),
                (1, format!("cuda:1 {}", nvidia_smi_field(1, "name"))),
            ],
            digests: [
                (POD_LEG_SEED, digest_a.clone()),
                (POD_LEG_SEED, digest_b.clone()),
            ],
            per_step_loss_delta: if deltas.is_empty() {
                vec![worst]
            } else {
                deltas.clone()
            },
            verdict_pass,
            reason: reason.clone(),
        };
        let artifact = build_gang_pod_artifact(inputs);

        // Written on BOTH the pass and the fail arm, BEFORE the assertion
        // below that would otherwise fail this test first — a failed run's
        // own numbers are exactly the evidence that has to survive.
        if let Some(dir) = &artifact_dir {
            let path = write_gang_pod_artifact(dir, &artifact).unwrap_or_else(|e| {
                panic!(
                    "{TEST}: verdict computed ({}) but failed to write the artifact to {}: {e}",
                    artifact.gang.verdict,
                    dir.display()
                )
            });
            tracing::info!(
                path = %path.display(),
                verdict = artifact.gang.verdict,
                git_sha,
                "gang pod-leg artifact written"
            );
        }

        assert!(
            verdict_pass,
            "{TEST} failed: {reason} (digest_a={digest_a}, digest_b={digest_b}, deltas={deltas:?})"
        );
    }
}

// ─── Hermetic oracle for the artifact's SHAPE (no GPU, no CUDA feature) ────

#[cfg(test)]
mod synthetic_artifact_tests {
    //! The only hermetic oracle for this artifact's shape (module doc,
    //! "Uncovered"): builds a synthetic `gang` pod-leg artifact with
    //! [`build_gang_pod_artifact`] — the SAME function the real pod run
    //! calls — from fixed inputs, and invokes
    //! `ci/scripts/check_cuda_run_artifacts.py`'s own `validate_artifact`
    //! against it (via a tiny importer script, never a re-implementation of
    //! the checker's rules).

    use super::{
        build_gang_pod_artifact, git_head_sha, write_gang_pod_artifact, GangPodArtifact,
        GangPodArtifactInputs, GANG_POD_LEG_EPSILON_REGISTERED_SHA,
    };

    /// Imports `ci/scripts/check_cuda_run_artifacts.py` by path and calls
    /// its `validate_artifact` directly against the given artifact — the
    /// SAME function `run_gate` calls per committed file.
    const CHECKER_INVOKER_PY: &str = r#"
import sys, json, importlib.util
from pathlib import Path

repo_root = Path(sys.argv[1])
artifact_path = Path(sys.argv[2])
relpath = sys.argv[3]

spec = importlib.util.spec_from_file_location(
    "check_cuda_run_artifacts", str(repo_root / "ci" / "scripts" / "check_cuda_run_artifacts.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

data = json.loads(artifact_path.read_text())
tracked = mod.git_ls_files(repo_root)
findings = mod.validate_artifact(data, relpath, repo_root, tracked, {})
print(json.dumps(findings))
"#;

    /// Run the checker's `validate_artifact` (via [`CHECKER_INVOKER_PY`])
    /// against `artifact`, and return its findings — empty means clean.
    fn run_checker(artifact: &serde_json::Value, relpath: &str) -> Vec<String> {
        let repo_root = super::workspace_root();
        let tmp = tempfile::tempdir().unwrap();
        let artifact_path = tmp.path().join("artifact.json");
        std::fs::write(&artifact_path, serde_json::to_vec_pretty(artifact).unwrap()).unwrap();
        let script_path = tmp.path().join("run_check.py");
        std::fs::write(&script_path, CHECKER_INVOKER_PY).unwrap();
        let output = std::process::Command::new("python3")
            .arg(&script_path)
            .arg(&repo_root)
            .arg(&artifact_path)
            .arg(relpath)
            .output()
            .expect(
                "failed to run python3 — required for this hermetic oracle (the checker itself \
                 is a python script)",
            );
        assert!(
            output.status.success(),
            "the checker invocation itself failed (not a validation finding): stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
            panic!("the checker did not print a JSON findings list: {e}: {stdout:?}")
        })
    }

    /// Like [`run_checker`], but writes `artifact` through
    /// [`write_gang_pod_artifact`] — the SAME writer the real pod run
    /// calls — rather than a hand-serialized JSON file, so this exercises
    /// the writer's OWN filename convention and I/O, not just
    /// [`build_gang_pod_artifact`]'s in-memory shape.
    fn run_checker_written(artifact: &GangPodArtifact) -> Vec<String> {
        let repo_root = super::workspace_root();
        let tmp = tempfile::tempdir().unwrap();
        let path = write_gang_pod_artifact(tmp.path(), artifact)
            .unwrap_or_else(|e| panic!("write_gang_pod_artifact failed: {e}"));
        let relpath = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("a written artifact always has a UTF-8 filename")
            .to_string();
        let script_path = tmp.path().join("run_check.py");
        std::fs::write(&script_path, CHECKER_INVOKER_PY).unwrap();
        let output = std::process::Command::new("python3")
            .arg(&script_path)
            .arg(&repo_root)
            .arg(&path)
            .arg(&relpath)
            .output()
            .expect(
                "failed to run python3 — required for this hermetic oracle (the checker itself \
                 is a python script)",
            );
        assert!(
            output.status.success(),
            "the checker invocation itself failed (not a validation finding): stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
            panic!("the checker did not print a JSON findings list: {e}: {stdout:?}")
        })
    }

    /// A clean, well-formed pod-leg `pass` artifact from fixed inputs — the
    /// SAME writer the real pod run calls.
    fn clean_pass_artifact() -> GangPodArtifact {
        let inputs = GangPodArtifactInputs {
            git_sha: git_head_sha(),
            box_name: "a100-fixture".to_string(),
            ranks: vec![
                (0, "cuda:0 NVIDIA A100-SXM4-80GB".to_string()),
                (1, "cuda:1 NVIDIA A100-SXM4-80GB".to_string()),
            ],
            digests: [(4242, "a".repeat(64)), (4242, "a".repeat(64))],
            per_step_loss_delta: vec![0.0, 1.0e-7, 2.0e-7],
            verdict_pass: true,
            reason: String::new(),
        };
        build_gang_pod_artifact(inputs)
    }

    /// EXECUTED ORACLE: the writer's own clean pass-shape output, written
    /// through [`write_gang_pod_artifact`], satisfies rule (k) cleanly.
    #[test]
    fn synthetic_pass_artifact_satisfies_rule_k() {
        let findings = run_checker_written(&clean_pass_artifact());
        assert!(
            findings.is_empty(),
            "the writer's own clean pass artifact must satisfy rule (k), got {findings:?}"
        );
    }

    /// A well-formed `fail` artifact (reason set, status RED, an unequal
    /// digest pair) must ALSO satisfy rule (k) cleanly — the module doc's
    /// "a fail is written, never suppressed": the shape must admit a
    /// failing run, not only a passing one.
    #[test]
    fn synthetic_fail_artifact_satisfies_rule_k() {
        let inputs = GangPodArtifactInputs {
            git_sha: git_head_sha(),
            box_name: "a100-fixture".to_string(),
            ranks: vec![
                (0, "cuda:0 NVIDIA A100-SXM4-80GB".to_string()),
                (1, "cuda:1 NVIDIA A100-SXM4-80GB".to_string()),
            ],
            digests: [(4242, "a".repeat(64)), (4242, "b".repeat(64))],
            per_step_loss_delta: vec![0.0, 5.0e-2],
            verdict_pass: false,
            reason: "the same-seed W=2 digest pair disagreed".to_string(),
        };
        let artifact = build_gang_pod_artifact(inputs);
        let findings = run_checker_written(&artifact);
        assert!(
            findings.is_empty(),
            "a well-formed fail artifact must ALSO satisfy rule (k) — a fail is admitted, not \
             refused, got {findings:?}"
        );
    }

    /// EXECUTED MUTATION: dropping `gang.epsilon` from an otherwise-clean
    /// artifact must be NAMED by the checker — the non-vacuous half of the
    /// oracle above (a checker that never fires would pass the clean case
    /// vacuously). Applied by hand against this test during authoring
    /// (removed `epsilon` from [`build_gang_pod_artifact`]'s output,
    /// confirmed the finding contains `gang.epsilon`, reverted) and pinned
    /// here so the mutation stays executed on every run, not only once.
    #[test]
    fn missing_epsilon_is_named_by_the_checker() {
        let mut artifact = serde_json::to_value(clean_pass_artifact()).unwrap();
        artifact
            .get_mut("gang")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("epsilon");
        let findings = run_checker(&artifact, "x.json");
        assert!(
            findings.iter().any(|f| f.contains("gang.epsilon")),
            "removing gang.epsilon must be named by the checker, got {findings:?}"
        );
    }

    /// The registered ε's own registration commit must be well-formed
    /// (40-hex) — a standing shape check, separate from the ancestry claim
    /// [`synthetic_pass_artifact_satisfies_rule_k`] already exercises
    /// through the real checker (that test would go red on its own if this
    /// constant were malformed or unreachable from HEAD).
    #[test]
    fn epsilon_registered_sha_is_well_formed() {
        assert_eq!(
            GANG_POD_LEG_EPSILON_REGISTERED_SHA.len(),
            40,
            "the registered epsilon sha must be 40-hex"
        );
        assert!(
            GANG_POD_LEG_EPSILON_REGISTERED_SHA
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "the registered epsilon sha must be lowercase hex"
        );
    }
}
