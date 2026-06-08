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
