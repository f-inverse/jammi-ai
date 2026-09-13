//! The persisted, self-describing training specification.
//!
//! A training job is durable work claimed under a lease (`training_jobs`). For a
//! worker to run a job it never submitted — on a fresh process, with no
//! in-memory carryover from the submitting session — the job must carry
//! *everything* needed to reconstruct its data and model. [`TrainingSpec`] is
//! that self-contained description: a tagged enum whose variant names the verb
//! that produced the job (`kind`) and whose fields are the reconstruction inputs
//! (source SQL, column bindings, deterministic sampler seeds, the architecture /
//! objective / optimisation budget).
//!
//! The spec is serialised to JSON into `training_jobs.training_spec` at submit
//! time and deserialised by the worker at claim time; the variant tag is also
//! written into `training_jobs.kind`. The same three verbs that submit
//! (`fine_tune`, `fine_tune_graph`, `train_context_predictor`) are the three
//! variants here — the unification the worker descends over once, rather than
//! three bespoke execution paths.

use serde::{Deserialize, Serialize};

use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;

use crate::fine_tune::graph_sampler::{GraphFineTuneSources, GraphSampleConfig};
use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
use crate::pipeline::context_predictor::ContextPredictorTrainConfig;

/// A durable, self-contained description of a training job: the verb that
/// produced it (the variant) and the inputs a worker reconstructs the run from
/// on a fresh process. Persisted as JSON on the job's `training_spec` column;
/// the variant's [`TrainingSpec::kind`] tag is mirrored into `training_jobs.kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrainingSpec {
    /// Column-source contrastive / classification / regression fine-tune. The
    /// worker re-runs `SELECT columns FROM source` and rebuilds the data loader
    /// keyed by `task` — the same loader the submitting `fine_tune` would have
    /// built, but from the persisted column bindings rather than in-memory
    /// batches.
    FineTune {
        /// Registered source whose rows the model is fine-tuned on.
        source: String,
        /// Columns selected into the training loader (the format — contrastive /
        /// triplet / pairs / classification — is detected from these names).
        columns: Vec<String>,
        /// Adapter method.
        method: FineTuneMethod,
        /// Task the fine-tuned model performs (selects the loader's cell
        /// decoding and the training head).
        task: ModelTask,
        /// Common base-model + optimisation knobs.
        common: TrainingCommon,
    },
    /// Graph-supervised fine-tune. The worker re-reads the node/edge sources and
    /// rebuilds the [`crate::fine_tune::graph_sampler::GraphSampler`] from
    /// `sample_config` — deterministic via its seed, so the re-sampled pairs
    /// match a run from the same spec byte-for-byte.
    GraphFineTune {
        /// Node-text + edge sources and their column bindings.
        sources: GraphFineTuneSources,
        /// Walk / negative-sampling knobs, including the deterministic seed.
        sample_config: GraphSampleConfig,
        /// Common base-model + optimisation knobs.
        common: TrainingCommon,
    },
    /// Episodic in-context-predictor meta-training. The worker re-samples the
    /// episodic meta-dataset from `source` per `predictor_spec` (deterministic
    /// task split via the spec seed), drives the predictor train loop, and
    /// persists the trained predictor.
    ContextPredictor {
        /// Source whose embedding table the episodic context is sampled from.
        source: String,
        /// The architecture / objective / episodic / optimisation specification.
        predictor_spec: ContextPredictorTrainConfig,
    },
}

/// Base-model and optimisation knobs common to the two LoRA fine-tune kinds. The
/// predictor kind carries its budget inside its own `predictor_spec`, so this is
/// shared only by the fine-tune variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCommon {
    /// Base model id the adapter / head is trained over.
    pub base_model: String,
    /// Fine-tune configuration (epochs, LoRA rank, loss, …).
    pub config: FineTuneConfig,
    /// How many data-parallel ranks this job trains over.
    /// [`DEFAULT_WORLD_SIZE`] is the single-rank, single-process run.
    ///
    /// Identity-relevant, not a scheduling hint: the rank count fixes the
    /// batch layout and the gradient summation order, so it belongs in the
    /// persisted spec beside the other reconstruction inputs rather than in
    /// the deployment's configuration. A count the deployment cannot serve
    /// is refused at the submit edge with a typed error, never clamped: a
    /// silently lowered count would train a different model than the caller
    /// asked for and record it under the same identity.
    ///
    /// A spec whose JSON carries no count deserializes to
    /// [`DEFAULT_WORLD_SIZE`], so a queued job written by any writer runs on
    /// any worker. Always serialized (no `skip_serializing_if`): the
    /// persisted JSON states the count rather than leaving a reader to infer
    /// it from an absence.
    #[serde(default = "default_world_size")]
    pub world_size: u32,
}

/// The rank count of a spec that does not name one: a single rank, which is
/// the single-process run.
pub const DEFAULT_WORLD_SIZE: u32 = 1;

/// `serde`'s `default` hook for [`TrainingCommon::world_size`] — a function
/// because `#[serde(default = ...)]` names a path, not a literal.
fn default_world_size() -> u32 {
    DEFAULT_WORLD_SIZE
}

/// What a submitted [`TrainingSpec`] is admitted against: the deployment's
/// devices, the collective it reduces over, and whether this build can reach
/// that collective.
///
/// The submit edge is where a rank count the deployment cannot serve is
/// caught, because it is the last point at which refusing costs nothing: past
/// it the spec is a durable row that a worker will claim, fail, and retry
/// until the attempt budget runs out. A count is never CLAMPED to what the
/// deployment can serve — a silently lowered count trains a different model
/// than the caller asked for and records it under the same identity.
///
/// The build flag is DATA rather than a `cfg!` inside the check so the rule
/// is decidable for either build from either build: a host test can state
/// what a CUDA build admits, and a CUDA build can state what a host build
/// refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankAdmission {
    devices: usize,
    collective: jammi_db::config::CollectiveSelection,
    cuda_build: bool,
}

impl RankAdmission {
    /// The admission this deployment implies: one rank per configured
    /// device, the configured collective, and THIS build's CUDA support.
    pub fn from_config(config: &jammi_db::config::JammiConfig) -> Self {
        Self {
            devices: config.gpu.device_list().len(),
            collective: config.worker.collective,
            cuda_build: cfg!(feature = "cuda"),
        }
    }

    /// An admission stated outright — for a test that needs a deployment
    /// this host does not have.
    pub fn new(
        devices: usize,
        collective: jammi_db::config::CollectiveSelection,
        cuda_build: bool,
    ) -> Self {
        Self {
            devices,
            collective,
            cuda_build,
        }
    }

    /// Admit `spec`, or refuse with a typed [`JammiError::Config`] naming the
    /// bound it crosses.
    ///
    /// The refusals, each of which would otherwise become a job that fails
    /// on a worker rather than at the caller:
    ///
    /// - `world_size == 0` — no rank to run on. Unrepresentable at the
    ///   request edge (`FineTuneRequest::world_size` is a `NonZeroU32`) but
    ///   reachable through a hand-built or deserialized spec, so the durable
    ///   edge checks it too.
    /// - `world_size > devices` — there is no device for the last rank, and
    ///   the alternative to refusing is two ranks silently sharing one.
    /// - `collective = "nccl"` on a build without CUDA — the requested
    ///   collective cannot be reached. Also refused at session OPEN
    ///   (`refuse_unreachable_collective`), which is the edge a live session
    ///   is stopped at; this arm is the same rule stated where the spec is
    ///   admitted, so a caller holding an admission built by hand gets the
    ///   same verdict.
    /// - `world_size > 1` with GradCache (`config.cached`) or hard-negative
    ///   mining (`config.hard_negatives.mine`) — both are single-rank
    ///   mechanisms (a whole-batch second pass, and a miner over this
    ///   process's own index), and running them per rank would change the
    ///   negative pool each rank sees, so the gang would not compute the
    ///   objective the caller asked for.
    ///
    /// The context-predictor kind carries no rank count at all: its variant
    /// has no [`TrainingCommon`], so a multi-rank predictor job is
    /// unrepresentable here and is refused at the wire decode, the last edge
    /// that can still see a count a caller chose.
    pub fn admit(&self, spec: &TrainingSpec) -> Result<()> {
        let common = match spec {
            TrainingSpec::FineTune { common, .. } | TrainingSpec::GraphFineTune { common, .. } => {
                common
            }
            TrainingSpec::ContextPredictor { .. } => return Ok(()),
        };
        let world_size = common.world_size;

        if world_size == 0 {
            return Err(JammiError::Config(
                "world_size must be >= 1 (1 is the single-rank job; 0 has no rank to run on)"
                    .into(),
            ));
        }
        if world_size as usize > self.devices {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} exceeds the {} configured device(s): one rank per                  device, so list more in `[gpu] devices` or submit a smaller rank count",
                self.devices
            )));
        }
        if self.collective.requires_cuda() && !self.cuda_build {
            return Err(JammiError::Config(format!(
                "[worker] collective = \"{}\" needs a build with the `cuda` feature; this                  binary has none, so the requested collective cannot be reached",
                self.collective
            )));
        }
        if world_size > 1 && common.config.cached {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} cannot be combined with GradCache (`cached`): the                  cached objective's second pass is over the WHOLE batch on one rank, so a                  gang would not compute the objective this config asks for"
            )));
        }
        if world_size > 1 && common.config.hard_negatives.mine {
            return Err(JammiError::Config(format!(
                "world_size = {world_size} cannot be combined with hard-negative mining                  (`hard_negatives.mine`): the miner retrieves from this process's own index,                  so each rank would mine a different negative pool"
            )));
        }
        Ok(())
    }
}

impl TrainingSpec {
    /// The catalog `kind` tag for this spec — the same discriminator the tagged
    /// JSON carries, mirrored into the `training_jobs.kind` column so a query can
    /// filter by verb without parsing the spec blob.
    pub fn kind(&self) -> &'static str {
        match self {
            TrainingSpec::FineTune { .. } => "fine_tune",
            TrainingSpec::GraphFineTune { .. } => "graph_fine_tune",
            TrainingSpec::ContextPredictor { .. } => "context_predictor",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JSON shape a writer that has no rank count queues: a `fine_tune`
    /// spec whose `common` block carries `base_model` and `config` and
    /// nothing else.
    ///
    /// The spec and the `common` block are built member by member rather than
    /// by serializing a current `TrainingSpec` and deleting a key — a fixture
    /// derived from the current shape would silently acquire whatever is
    /// added to `TrainingCommon` next and stop testing the absence it exists
    /// to test. Only `config`, an opaque block this test says nothing about,
    /// is taken from its own serializer.
    fn spec_json_without_a_rank_count() -> String {
        serde_json::json!({
            "kind": "fine_tune",
            "source": "patents",
            "columns": ["abstract"],
            "method": "lora",
            "task": "text_embedding",
            "common": {
                "base_model": "local:tiny",
                "config": serde_json::to_value(FineTuneConfig::default()).expect("config"),
            },
        })
        .to_string()
    }

    /// A spec whose JSON names no rank count deserializes to the single-rank
    /// run, so a job queued without the field runs on a worker that reads it.
    /// The alternative — a deserialization failure, or a `0` default — turns
    /// every already-queued job into a permanently unclaimable row.
    #[test]
    fn a_spec_with_no_rank_count_deserializes_to_a_single_rank() {
        let json = spec_json_without_a_rank_count();
        assert!(
            !json.contains("world_size"),
            "the fixture must not name the count it exists to omit: {json}"
        );
        let spec: TrainingSpec = serde_json::from_str(&json)
            .expect("a spec that names no rank count must still deserialize");
        let TrainingSpec::FineTune { common, .. } = &spec else {
            panic!("expected the fine_tune variant, got {spec:?}");
        };
        assert_eq!(
            common.world_size, 1,
            "an absent rank count is one rank, not zero and not a parse error"
        );
        assert_eq!(common.world_size, DEFAULT_WORLD_SIZE);
    }

    /// The count survives a round trip through the persisted form, and is
    /// written under the `world_size` key ALWAYS — a reader of `jobs.spec`
    /// (and the remote-versus-embedded parity oracle in the server suite)
    /// reads the count off the JSON rather than inferring it from an absence.
    #[test]
    fn a_chosen_rank_count_round_trips_and_is_always_serialized() {
        let spec = TrainingSpec::FineTune {
            source: "patents".into(),
            columns: vec!["abstract".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig::default(),
                world_size: 4,
            },
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        assert!(
            json.contains(r#""world_size":4"#),
            "the chosen count must be in the persisted JSON: {json}"
        );

        let single = TrainingSpec::FineTune {
            common: TrainingCommon {
                base_model: "local:tiny".into(),
                config: FineTuneConfig::default(),
                world_size: DEFAULT_WORLD_SIZE,
            },
            source: "patents".into(),
            columns: vec!["abstract".into()],
            method: crate::fine_tune::FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
        };
        let single_json = serde_json::to_string(&single).expect("serialize");
        assert!(
            single_json.contains(r#""world_size":1"#),
            "the default count is serialized too, never skipped: {single_json}"
        );

        let back: TrainingSpec = serde_json::from_str(&json).expect("deserialize");
        let TrainingSpec::FineTune { common, .. } = &back else {
            panic!("expected the fine_tune variant, got {back:?}");
        };
        assert_eq!(common.world_size, 4);
    }
}
