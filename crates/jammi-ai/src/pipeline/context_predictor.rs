//! Episodic meta-training for the amortized in-context predictor family
//! ([`jammi_encoders::AnyContextPredictor`]).
//!
//! A neural process is *meta-learned*: training samples a **task**, splits it
//! into a context set and a held-out target, and maximises the target's
//! predictive log-likelihood given the context — so adaptation to a new target's
//! neighbourhood is amortized into the weights and needs no gradient update at
//! inference. This module is that path. It owns three concerns and nothing else:
//!
//! - the [`ContextPredictorTrainConfig`] — the architecture/objective/episodic
//!   knobs, distinct from the text-row-shaped fine-tune `TrainingFormat` (a
//!   context-set→target model is not a text row, so it gets its own config);
//! - the **episodic sampler** — per target row, assemble its context through the
//!   S16 retrieval (`search` + `exclude_self` + a same-task split), read the
//!   members' x-vectors and y-labels **through the generic SQL surface**, and
//!   build a padded [`ContextEpisode`] plus the held-out target `y`. Tasks (not
//!   points) are partitioned into train/test, and a `min_task_count` guard
//!   rejects a degenerate meta-dataset;
//! - the trainer — build the predictor in a [`VarMap`], drive the generalized
//!   [`train_loop`] with the predictor's forward and S18's proper-scoring loss,
//!   persist the trained weights, and register the artifact in the catalog.
//!
//! ## Leakage (the HIGH context-leakage contract)
//!
//! A target must never appear in its own context, within or across episodes. The
//! sampler inherits S16's two guards verbatim: every per-target context request
//! sets `exclude_self` (dropping the target's own key) and scopes the context to
//! the target's own task split, so a target's own outcome is held out of every
//! episode's conditioning set.
//!
//! ## Held-out **task** split (the HIGH meta-overfitting contract)
//!
//! Generalisation is evaluated on held-out *tasks*, not held-out *points*: the
//! distinct values of the task column are partitioned, the predictor trains on
//! the train tasks' episodes, and the spec carries the test tasks so a caller
//! can measure the train-task→test-task generalisation gap. Too few tasks and
//! the predictor memorises rather than generalises, so a `min_task_count` guard
//! is a typed precondition rather than a silent degenerate run.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, StringArray};
use arrow::record_batch::RecordBatch;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;

use serde::{Deserialize, Serialize};

use jammi_encoders::{AnyContextPredictor, ContextEpisode, ContextPredictorConfig};

/// The curated in-context-predictor architecture a [`ContextPredictorTrainConfig`]
/// selects — re-exported so a consumer can name the architecture of the public
/// training config without depending on `jammi-encoders` directly.
pub use jammi_encoders::ContextArchitecture;

use crate::fine_tune::regression_loss;
use crate::fine_tune::spec::TrainingSpec;
use crate::inference::adapter::distribution::{DistributionAdapter, DistributionForm};
use crate::inference::adapter::{BackendOutput, OutputAdapter};
use crate::pipeline::context_set::{
    ContextRequest, ContextSource, ContextSourceKind, HybridMerge, SetAggregator,
};
use crate::pipeline::graph_neighbourhood::EdgeGather;
use crate::pipeline::parallel_train::{train_loop, ParallelTrainConfig};
use crate::predict::conformal::{ConformalModel, IntervalScore};
use crate::session::InferenceSession;

/// Shape of the predictive-distribution head the predictor emits and the
/// objective scores — the S18 output families, selected by config rather than by
/// a tensor op the caller writes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictiveHead {
    /// A 2-wide `(mean, raw_std)` Gaussian head.
    Gaussian {
        /// Proper score the Gaussian head trains against.
        objective: GaussianObjective,
    },
    /// A `levels`-wide quantile head trained with the pinball objective.
    Quantile {
        /// Ascending quantile levels in `(0, 1)`; the head width equals their
        /// count.
        levels: Vec<f64>,
    },
}

/// Which proper score a [`PredictiveHead::Gaussian`] head trains against — the
/// S18 objectives, reused, never reinvented here.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GaussianObjective {
    /// (β-)Gaussian negative log-likelihood; `beta` in `[0, 1]`, `0` the plain
    /// heteroscedastic NLL.
    Nll {
        /// β-NLL re-weighting exponent.
        beta: f64,
    },
    /// Closed-form Gaussian CRPS — the collapse-resistant, outcome-unit score.
    Crps,
}

impl PredictiveHead {
    /// The head width this output shape needs: `2` for a Gaussian
    /// `(mean, raw_std)`, `levels.len()` for a quantile head.
    fn head_width(&self) -> usize {
        match self {
            PredictiveHead::Gaussian { .. } => 2,
            PredictiveHead::Quantile { levels } => levels.len(),
        }
    }

    /// Score `preds` (`[B, head_width]`) against the held-out target `y` (`[B]`)
    /// with this head's S18 proper-scoring objective — the training loss, and
    /// the same objective a consumer evaluates a held-out-task generalisation
    /// gap with. No new loss code: dispatches to the `regression_loss` proper
    /// scores the fine-tune trainer also uses.
    pub fn score(&self, preds: &Tensor, target_y: &Tensor) -> Result<Tensor> {
        match self {
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Nll { beta },
            } => regression_loss::gaussian_nll_loss(preds, target_y, *beta),
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            } => regression_loss::crps_gaussian_loss(preds, target_y),
            PredictiveHead::Quantile { levels } => {
                regression_loss::pinball_loss(preds, target_y, levels)
            }
        }
    }

    /// Serialise the head form into the persisted config so a served predictor
    /// reconstructs the exact distribution shape (Gaussian vs the precise
    /// quantile levels), not merely the head width.
    fn to_config_json(&self) -> serde_json::Value {
        match self {
            PredictiveHead::Gaussian { .. } => serde_json::json!({ "kind": "gaussian" }),
            PredictiveHead::Quantile { levels } => {
                serde_json::json!({ "kind": "quantile", "levels": levels })
            }
        }
    }

    /// Reconstruct the served distribution form from a persisted head config (the
    /// `to_config_json` shape). Serving needs only the *form*, not the training
    /// objective, so a Gaussian head reloads its `(mean, std)` shape and a
    /// quantile head its exact levels.
    fn form_from_config(value: &serde_json::Value) -> Result<DistributionForm> {
        match value.get("kind").and_then(|k| k.as_str()) {
            Some("gaussian") => Ok(DistributionForm::Gaussian),
            Some("quantile") => {
                let levels = value
                    .get("levels")
                    .and_then(|l| l.as_array())
                    .ok_or_else(|| {
                        JammiError::Inference(
                            "context predictor config: quantile head missing levels".into(),
                        )
                    })?
                    .iter()
                    .map(|v| {
                        v.as_f64().ok_or_else(|| {
                            JammiError::Inference(
                                "context predictor config: quantile level is not a number".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<f64>>>()?;
                Ok(DistributionAdapter::quantile(levels)?.form().clone())
            }
            _ => Err(JammiError::Inference(
                "context predictor config: head form is neither gaussian nor quantile".into(),
            )),
        }
    }

    /// Reject a malformed quantile-level set up front (ascending, in `(0, 1)`),
    /// so a head width is never built against an incoherent objective.
    fn validate(&self) -> Result<()> {
        if let PredictiveHead::Quantile { levels } = self {
            if levels.is_empty() {
                return Err(JammiError::FineTune(
                    "Quantile head requires at least one level".into(),
                ));
            }
            if levels.iter().any(|&q| !(0.0..1.0).contains(&q)) {
                return Err(JammiError::FineTune(
                    "Quantile levels must lie strictly in (0, 1)".into(),
                ));
            }
            if levels.windows(2).any(|w| w[1] <= w[0]) {
                return Err(JammiError::FineTune(
                    "Quantile levels must be strictly ascending".into(),
                ));
            }
        }
        Ok(())
    }
}

/// The episodic meta-training specification — the architecture, the context
/// retrieval, the objective, and the optimisation budget for one
/// [`InferenceSession::train_context_predictor`] run.
///
/// A distinct type from the text-row fine-tune config: every consumer of the
/// fine-tune `TrainingFormat` is text-shaped, and a context-set→target episode
/// is not a text row. The episodic knobs live here, with the pipeline that uses
/// them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPredictorTrainConfig {
    /// The model id the trained predictor registers under in the catalog.
    pub model_id: String,
    /// Which curated [`AnyContextPredictor`] member to train.
    pub architecture: ContextArchitecture,
    /// The source's key column — the per-row identity episodes are sampled by
    /// and the context retrieval excludes self against.
    pub key_column: String,
    /// The source column whose distinct values are the **tasks** (a cohort, a
    /// time-window, a source partition). Episodes never mix tasks, and tasks —
    /// not points — are partitioned into train/test.
    pub task_column: String,
    /// The source column carrying the scalar outcome `y` the predictor regresses.
    pub value_column: String,
    /// Context size `k`: the retrieval over-fetches and pads the dense episode
    /// to this width.
    pub context_k: usize,
    /// Hidden width of the predictor's MLPs / transformer model dimension.
    pub hidden_dim: usize,
    /// Attention heads (AttnCnp, Tnp). Must divide `hidden_dim`.
    pub num_heads: usize,
    /// Transformer layers (Tnp).
    pub num_layers: usize,
    /// The predictive-distribution head + its proper-scoring objective.
    pub head: PredictiveHead,
    /// Passes over the sampled train episodes.
    pub epochs: usize,
    /// AdamW learning rate.
    pub learning_rate: f64,
    /// Global-L2 gradient-clip norm; `<= 0.0` disables clipping.
    pub grad_clip: f64,
    /// Fraction of the distinct tasks held out for the test split, in `(0, 1)`.
    pub test_task_fraction: f64,
    /// Minimum distinct task count the meta-dataset must carry — the
    /// meta-overfitting guard. A meta-dataset below this is rejected rather than
    /// run into memorisation.
    pub min_task_count: usize,
    /// Seed for the deterministic train/test **task** partition.
    pub seed: u64,
}

impl ContextPredictorTrainConfig {
    /// Validate the spec independently of any data: the head/objective is
    /// coherent, the test fraction is a proper fraction, `num_heads` divides
    /// `hidden_dim` for the attentive members, and the budget is non-degenerate.
    fn validate(&self) -> Result<()> {
        self.head.validate()?;
        if self.context_k == 0 {
            return Err(JammiError::FineTune("context_k must be at least 1".into()));
        }
        if self.epochs == 0 {
            return Err(JammiError::FineTune("epochs must be at least 1".into()));
        }
        if !(0.0..1.0).contains(&self.test_task_fraction) || self.test_task_fraction <= 0.0 {
            return Err(JammiError::FineTune(
                "test_task_fraction must lie strictly in (0, 1)".into(),
            ));
        }
        if self.min_task_count < 2 {
            return Err(JammiError::FineTune(
                "min_task_count must be at least 2 (one task each for train and test)".into(),
            ));
        }
        if matches!(
            self.architecture,
            ContextArchitecture::AttnCnp | ContextArchitecture::Tnp
        ) && (self.num_heads == 0 || self.hidden_dim % self.num_heads != 0)
        {
            return Err(JammiError::FineTune(format!(
                "num_heads ({}) must be non-zero and divide hidden_dim ({})",
                self.num_heads, self.hidden_dim
            )));
        }
        Ok(())
    }
}

/// One training batch the generalized [`train_loop`] drives: a padded
/// [`ContextEpisode`] (`B` targets, each with its assembled context) and the
/// held-out target outcomes `target_y` (`[B]`) the predictor's head is scored
/// against. The episodic peer of [`crate::pipeline::parallel_train::TensorBatch`].
#[derive(Debug, Clone)]
pub struct EpisodeBatch {
    /// The targets and their (padded, presence-masked) context sets.
    pub episode: ContextEpisode,
    /// Held-out target outcomes, `[B]` — the `y` each target's context-conditioned
    /// head predicts.
    pub target_y: Tensor,
}

/// The sampled meta-dataset: the train episodes the predictor optimises over and
/// the held-out **test** episodes a caller evaluates the generalisation gap on.
/// Tasks are disjoint across the two — no task contributes to both.
#[derive(Debug, Clone)]
pub struct SampledEpisodes {
    /// One [`EpisodeBatch`] per train task.
    pub train: Vec<EpisodeBatch>,
    /// One [`EpisodeBatch`] per held-out test task.
    pub test: Vec<EpisodeBatch>,
    /// Distinct task count the meta-dataset carried (train + test).
    pub task_count: usize,
}

/// One target row's assembled, leakage-scoped context — the members' x-vectors
/// and y-labels in retrieval order, plus the target's own `(x, y)`. The padded
/// [`ContextEpisode`] tensors are built by stacking these across a task.
struct TargetContext {
    target_x: Vec<f32>,
    target_y: f64,
    member_x: Vec<Vec<f32>>,
    member_y: Vec<f64>,
}

impl InferenceSession {
    /// Sample the episodic meta-dataset for `spec` over `source_id`'s embedding
    /// table, partitioning distinct **tasks** (not points) into train/test.
    ///
    /// For each target row in a task, the context is assembled through S16's
    /// retrieval with `exclude_self` and a same-task split (the leakage
    /// contract), the surviving members' x-vectors and y-labels are read back
    /// **through the generic SQL surface**, and a padded [`ContextEpisode`] plus
    /// the held-out target `y` are built. The task's targets become one
    /// [`EpisodeBatch`].
    ///
    /// The split is over tasks: the distinct `task_column` values are shuffled
    /// deterministically by `spec.seed` and the trailing `test_task_fraction`
    /// become the held-out tasks. A meta-dataset with fewer than
    /// `spec.min_task_count` distinct tasks is rejected.
    pub async fn sample_context_episodes(
        self: &Arc<Self>,
        source_id: &str,
        spec: &ContextPredictorTrainConfig,
    ) -> Result<SampledEpisodes> {
        spec.validate()?;

        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
            .await?;
        let feature_dim = table.dimensions.ok_or_else(|| {
            JammiError::FineTune(format!(
                "source '{source_id}' embedding table carries no vector dimension"
            ))
        })? as usize;

        let mut tasks = self.distinct_tasks(source_id, &spec.task_column).await?;
        if tasks.len() < spec.min_task_count {
            return Err(JammiError::FineTune(format!(
                "meta-dataset has {} distinct task(s) in column '{}', below the \
                 min_task_count of {} — too few to meta-train without memorising",
                tasks.len(),
                spec.task_column,
                spec.min_task_count
            )));
        }
        let task_count = tasks.len();
        deterministic_shuffle(&mut tasks, spec.seed);
        let n_test = (((task_count as f64) * spec.test_task_fraction).ceil() as usize)
            .clamp(1, task_count - 1);
        let test_tasks = tasks.split_off(task_count - n_test);
        let train_tasks = tasks;

        let device = crate::model::backend::candle::select_device(self.device_config());

        let train = self
            .episodes_for_tasks(source_id, &table, spec, &train_tasks, feature_dim, &device)
            .await?;
        let test = self
            .episodes_for_tasks(source_id, &table, spec, &test_tasks, feature_dim, &device)
            .await?;

        Ok(SampledEpisodes {
            train,
            test,
            task_count,
        })
    }

    /// Submit an episodic in-context-predictor meta-training job on
    /// `source_id`'s embedding table per `spec`, returning a
    /// [`TrainingJob`](crate::fine_tune::training_job::TrainingJob) handle
    /// immediately.
    ///
    /// Like the fine-tune verbs, this persists a self-describing
    /// [`TrainingSpec::ContextPredictor`] into a `queued` catalog job and
    /// returns; a [`crate::fine_tune::worker::TrainingWorker`] later claims it,
    /// re-samples the episodic meta-dataset from the persisted spec
    /// (deterministic task split via the seed), drives the predictor train loop
    /// while heartbeating, and registers the trained predictor. Call
    /// `job.wait().await` to block until a worker drives it to completion;
    /// `job.model_id()` is the model id the predictor registers under (the
    /// spec's `model_id`).
    pub async fn train_context_predictor(
        self: &Arc<Self>,
        source_id: &str,
        spec: &ContextPredictorTrainConfig,
    ) -> Result<crate::fine_tune::training_job::TrainingJob> {
        spec.validate()?;

        // The predictor registers under its own model id; the base-model FK on
        // the job points at the source's embedding model so the row is valid.
        // The table records the model's bare name; ensure a catalog row exists
        // for it (an embedding table can be materialised without registering a
        // model row) and use its PK (`name::version`) for the FK.
        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
            .await?;
        let base_model_pk = match self.catalog().get_model(&table.model_id).await? {
            Some(m) => crate::model::to_catalog_pk(&m.model_id, m.version),
            None => {
                self.catalog()
                    .register_model(RegisterModelParams {
                        model_id: &table.model_id,
                        version: 1,
                        model_type: "embedding",
                        backend: "candle",
                        task: ModelTask::TextEmbedding,
                        base_model_id: None,
                        artifact_path: None,
                        config_json: None,
                    })
                    .await?;
                crate::model::to_catalog_pk(&table.model_id, 1)
            }
        };
        let loss_type = format!("{:?}", spec.head);
        let hyperparams = serde_json::to_string(spec)?;

        let job_id = uuid::Uuid::new_v4().to_string();
        let training_spec = TrainingSpec::ContextPredictor {
            source: source_id.to_string(),
            predictor_spec: spec.clone(),
        };
        let spec_json = serde_json::to_string(&training_spec)?;
        self.catalog()
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: &job_id,
                base_model_id: &base_model_pk,
                training_source: source_id,
                loss_type: &loss_type,
                hyperparams: &hyperparams,
                kind: training_spec.kind(),
                training_spec: &spec_json,
            })
            .await?;

        Ok(crate::fine_tune::training_job::TrainingJob::new(
            job_id,
            "queued".into(),
            spec.model_id.clone(),
            Arc::clone(self.catalog_arc()),
        ))
    }

    /// Run an in-context-predictor meta-training to completion: sample the
    /// episodic meta-dataset, build the predictor the spec selects in a fresh
    /// [`VarMap`], drive the generalized [`train_loop`] (checking `cancel` at
    /// every epoch boundary), and write the trained weights into a local tempdir
    /// described by a `ModelTask::Regression` registration — the same
    /// model-artifact path the fine-tune trainer produces its output through.
    ///
    /// This is the worker-side execution the [`Self::train_context_predictor`]
    /// submit path defers to; it reconstructs everything from `spec` alone, with
    /// no in-memory carryover from the submitting session.
    ///
    /// It does **not** publish the artifact or register the catalog row — the
    /// worker's unified finalize does both (publishing under a unique per-attempt
    /// prefix, registering through its tenant-pinned catalog), so the predictor
    /// path takes the same catalog-pointer-as-commit route as the fine-tune
    /// kinds. The sampling reads off the session (`resolve_embedding_table`, the
    /// SQL-surface context assembly and per-member vector reads) are
    /// tenant-scoped: the worker runs this whole call inside the job's tenant
    /// scope, so the session's shared tenant binding resolves to the job's tenant
    /// rather than the worker's unbound default.
    pub(crate) async fn run_context_predictor_training(
        self: &Arc<Self>,
        source_id: &str,
        spec: &ContextPredictorTrainConfig,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<crate::fine_tune::worker::TrainedArtifact> {
        let sampled = self.sample_context_episodes(source_id, spec).await?;
        if sampled.train.is_empty() {
            return Err(JammiError::FineTune(
                "no train tasks survived the meta-dataset split".into(),
            ));
        }

        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
            .await?;
        let feature_dim = table.dimensions.ok_or_else(|| {
            JammiError::FineTune(format!(
                "source '{source_id}' embedding table carries no vector dimension"
            ))
        })? as usize;
        let device = crate::model::backend::candle::select_device(self.device_config());

        let varmap = VarMap::new();
        let predictor = build_predictor(spec, feature_dim, &varmap, &device)?;

        let train_config = ParallelTrainConfig {
            epochs: spec.epochs,
            learning_rate: spec.learning_rate,
            weight_decay: 0.0,
            grad_clip: spec.grad_clip,
        };
        train_loop(
            &varmap,
            &sampled.train,
            &train_config,
            cancel,
            |batch: &EpisodeBatch| {
                predictor
                    .forward(&batch.episode)
                    .map_err(|e| JammiError::FineTune(format!("context predictor forward: {e}")))
            },
            |preds, batch: &EpisodeBatch| spec.head.score(preds, &batch.target_y),
        )?;

        self.persist_predictor(spec, &table, &varmap)
    }

    /// Build one [`EpisodeBatch`] per task — every target row in the task, each
    /// with its leakage-scoped context assembled and read back through SQL.
    async fn episodes_for_tasks(
        self: &Arc<Self>,
        source_id: &str,
        table: &ResultTableRecord,
        spec: &ContextPredictorTrainConfig,
        tasks: &[String],
        feature_dim: usize,
        device: &Device,
    ) -> Result<Vec<EpisodeBatch>> {
        let mut batches = Vec::with_capacity(tasks.len());
        for task in tasks {
            batches.push(
                self.episode_for_task(source_id, table, spec, task, feature_dim, device)
                    .await?,
            );
        }
        Ok(batches)
    }

    /// Build one task's [`EpisodeBatch`]: every target row in `task`, each with
    /// its leakage-scoped context assembled and read back through SQL, padded to
    /// the configured `context_k`.
    async fn episode_for_task(
        self: &Arc<Self>,
        source_id: &str,
        table: &ResultTableRecord,
        spec: &ContextPredictorTrainConfig,
        task: &str,
        feature_dim: usize,
        device: &Device,
    ) -> Result<EpisodeBatch> {
        let targets = self.task_targets(source_id, spec, task).await?;
        if targets.is_empty() {
            return Err(JammiError::FineTune(format!(
                "task '{task}' in column '{}' has no rows",
                spec.task_column
            )));
        }

        // The split predicate scopes the context to the target's own task, so a
        // context member is always a same-task neighbour and never spills across
        // the train/test task boundary.
        let split = format!(
            "arrow_cast(\"{}\", 'Utf8') = '{}'",
            spec.task_column,
            escape_sql_literal(task)
        );

        let mut contexts = Vec::with_capacity(targets.len());
        for (target_key, target_y) in &targets {
            // The target's own stored vector is the retrieval query — an ordinary
            // SQL-surface read, not a raw-vector verb.
            let target_x = self
                .read_vector_by_key(table, target_key)
                .await?
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "target '{target_key}' has no stored vector in '{}'",
                        table.table_name
                    ))
                })?;

            // S16 retrieval, leakage-scoped: exclude_self drops the target's own
            // key, the split keeps the context inside the same task — so the
            // target's outcome can never enter its own context, in or across
            // episodes. value_columns hydrates the members' y-labels in key order.
            let mut request = ContextRequest::new(source_id, target_x.clone(), spec.context_k);
            request.exclude_self = true;
            request.exclude_key = Some(target_key.clone());
            request.split = Some(split.clone());
            request.aggregator = SetAggregator::Mean;
            request.value_columns = vec![spec.value_column.clone()];
            let rep = self.assemble_context(&request).await?;

            // Per-member x-vectors via the generic SQL surface, keyed by the
            // leakage-scoped member keys assemble_context surfaced (same order as
            // the hydrated y-labels).
            let member_x = self.read_member_vectors(table, &rep.context_keys).await?;
            let member_y = extract_value_column(&rep.value_rows, &spec.value_column)?;
            if member_x.len() != member_y.len() {
                return Err(JammiError::FineTune(format!(
                    "context member x/y count mismatch for target '{target_key}': \
                     {} vectors vs {} labels",
                    member_x.len(),
                    member_y.len()
                )));
            }

            contexts.push(TargetContext {
                target_x,
                target_y: *target_y,
                member_x,
                member_y,
            });
        }

        let episode = pad_episode(&contexts, feature_dim, spec.context_k, device)?;
        let target_y = Tensor::from_vec(
            contexts
                .iter()
                .map(|c| c.target_y as f32)
                .collect::<Vec<f32>>(),
            (contexts.len(),),
            device,
        )
        .map_err(|e| JammiError::FineTune(format!("target_y tensor: {e}")))?;

        Ok(EpisodeBatch { episode, target_y })
    }

    /// Read the stored `vector` of each key from the embedding table, in the
    /// given (retrieval) order, through the generic SQL surface — a typed
    /// `_row_id IN (keys)` scan of `jammi.{table}`, the per-member read the
    /// attentive members attend over. The keys are bound IN-list values, never
    /// interpolated, so an arbitrary key is not an injection vector.
    async fn read_member_vectors(
        &self,
        table: &ResultTableRecord,
        context_keys: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        if context_keys.is_empty() {
            return Ok(Vec::new());
        }
        use datafusion::prelude::{col, lit};

        let table_ref =
            datafusion::sql::TableReference::bare(format!("jammi.{}", table.table_name));
        let keys: Vec<datafusion::prelude::Expr> =
            context_keys.iter().map(|k| lit(k.as_str())).collect();
        let batches = self
            .context()
            .table(table_ref.clone())
            .await
            .map_err(|e| {
                JammiError::FineTune(format!("member-vectors resolve '{table_ref}': {e}"))
            })?
            .filter(col("_row_id").in_list(keys, false))
            .map_err(|e| JammiError::FineTune(format!("member-vectors filter: {e}")))?
            .select_columns(&["_row_id", "vector"])
            .map_err(|e| JammiError::FineTune(format!("member-vectors projection: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::FineTune(format!("member-vectors scan: {e}")))?;

        order_vectors_by_keys(&batches, &table.table_name, context_keys)
    }

    /// Distinct values of the task column, in scan order.
    async fn distinct_tasks(&self, source_id: &str, task_column: &str) -> Result<Vec<String>> {
        let source_table = self.find_table_name(source_id)?;
        let batches = self
            .context()
            .sql(&format!(
                "SELECT DISTINCT arrow_cast(\"{task_column}\", 'Utf8') AS _task \
                 FROM \"{source_id}\".public.\"{source_table}\" \
                 WHERE \"{task_column}\" IS NOT NULL ORDER BY _task"
            ))
            .await
            .map_err(|e| JammiError::FineTune(format!("distinct tasks scan: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::FineTune(format!("distinct tasks collect: {e}")))?;
        Ok(collect_string_column(&batches, "_task"))
    }

    /// The `(key, y)` of every row in one task, in scan order — the target rows
    /// the task's episode predicts. `key` and `y` are read with `arrow_cast` so
    /// an integer key or a float-typed outcome resolve to the `(Utf8, Float64)`
    /// the sampler tensors expect.
    async fn task_targets(
        &self,
        source_id: &str,
        spec: &ContextPredictorTrainConfig,
        task: &str,
    ) -> Result<Vec<(String, f64)>> {
        let source_table = self.find_table_name(source_id)?;
        let batches = self
            .context()
            .sql(&format!(
                "SELECT arrow_cast(\"{key}\", 'Utf8') AS _key, \
                        arrow_cast(\"{value}\", 'Float64') AS _y \
                 FROM \"{source_id}\".public.\"{source_table}\" \
                 WHERE arrow_cast(\"{task_col}\", 'Utf8') = '{task}'",
                key = spec.key_column,
                value = spec.value_column,
                task_col = spec.task_column,
                task = escape_sql_literal(task),
            ))
            .await
            .map_err(|e| JammiError::FineTune(format!("task targets scan: {e}")))?
            .collect()
            .await
            .map_err(|e| JammiError::FineTune(format!("task targets collect: {e}")))?;
        Ok(collect_key_value(&batches))
    }

    /// Write the trained predictor's weights into a local tempdir and describe
    /// the catalog model row, mirroring the fine-tune trainer's model-output
    /// path: the weights land as `model.safetensors`, and a
    /// `ModelTask::Regression` registration points the row at the artifact.
    ///
    /// The weights are written to a fresh worker-private tempdir (under the
    /// deployment's training scratch dir), so two workers running the same
    /// predictor job never share a training-time file. The returned
    /// [`crate::fine_tune::worker::TrainedArtifact`] is published to the artifact
    /// store under a unique per-attempt prefix and registered by the worker's
    /// unified finalize — the catalog-pointer-as-commit, identical to the
    /// fine-tune kinds.
    fn persist_predictor(
        &self,
        spec: &ContextPredictorTrainConfig,
        table: &ResultTableRecord,
        varmap: &VarMap,
    ) -> Result<crate::fine_tune::worker::TrainedArtifact> {
        let scratch = self.inner_config().artifact_dir.join("context_predictors");
        std::fs::create_dir_all(&scratch)?;
        let dir = tempfile::Builder::new()
            .prefix("predictor-")
            .tempdir_in(&scratch)?;
        let weights_path = dir.path().join("model.safetensors");

        let named: HashMap<String, Tensor> = {
            let data = varmap
                .data()
                .lock()
                .map_err(|_| JammiError::FineTune("predictor varmap lock poisoned".into()))?;
            data.iter()
                .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
                .collect()
        };
        candle_core::safetensors::save(&named, &weights_path)
            .map_err(|e| JammiError::FineTune(format!("save predictor weights: {e}")))?;

        let config_json = serde_json::json!({
            "architecture": format!("{:?}", spec.architecture),
            "context_k": spec.context_k,
            "feature_dim": table.dimensions,
            "hidden_dim": spec.hidden_dim,
            "num_heads": spec.num_heads,
            "num_layers": spec.num_layers,
            "head_width": spec.head.head_width(),
            "head": spec.head.to_config_json(),
            "value_column": spec.value_column,
        })
        .to_string();

        Ok(crate::fine_tune::worker::TrainedArtifact {
            dir,
            register: crate::fine_tune::worker::ModelRegistration {
                model_id: spec.model_id.clone(),
                model_type: "context-predictor",
                task: ModelTask::Regression,
                base_model_id: Some(table.model_id.clone()),
                config_json: Some(config_json),
            },
            metrics: None,
        })
    }
}

/// A served predictive distribution for one target — the trained head floats run
/// through the S18 [`DistributionAdapter`], so a Gaussian head yields a
/// `(mean, std)` and a quantile head an ascending `(level, value)` set. This is
/// exactly the per-row distribution the fine-tune regression serving path emits;
/// the only difference is the head was produced by an in-context forward over a
/// live context rather than a fixed encoder.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictedDistribution {
    /// Parametric Gaussian: the predictive mean and the served `σ` (floored
    /// softplus of the raw scale, the same transform the trainer's σ uses).
    Gaussian {
        /// The predictive mean `μ`.
        mean: f32,
        /// The served standard deviation `σ = floor + softplus(raw)`.
        std: f32,
    },
    /// Quantile: one `(level, value)` per declared level, ascending and
    /// non-crossing (the adapter sorts each row).
    Quantile {
        /// `(level, predicted_value)` pairs in ascending level order.
        levels: Vec<(f64, f32)>,
    },
}

impl PredictedDistribution {
    /// The predictive mean a conformal `AbsoluteResidual` wrap centres its
    /// interval on (Gaussian head), or the central quantile estimate (quantile
    /// head, the level nearest `0.5`).
    fn point_estimate(&self) -> f32 {
        match self {
            PredictedDistribution::Gaussian { mean, .. } => *mean,
            PredictedDistribution::Quantile { levels } => levels
                .iter()
                .min_by(|a, b| (a.0 - 0.5).abs().total_cmp(&(b.0 - 0.5).abs()))
                .map(|(_, v)| *v)
                .unwrap_or(0.0),
        }
    }

    /// The `(lower, upper)` quantile estimates a conformal `Cqr` wrap adjusts —
    /// the extreme served levels. Returns `None` for a Gaussian head (which is
    /// wrapped with `AbsoluteResidual`, not `Cqr`).
    fn quantile_bounds(&self) -> Option<(f32, f32)> {
        match self {
            PredictedDistribution::Gaussian { .. } => None,
            PredictedDistribution::Quantile { levels } => {
                let lo = levels.first().map(|(_, v)| *v)?;
                let hi = levels.last().map(|(_, v)| *v)?;
                Some((lo, hi))
            }
        }
    }
}

/// The live-context source a served predictor assembles each target's context
/// from. `Ann` reproduces S16 retrieval (using the trained `context_k`);
/// `Edges`/`Hybrid` re-gather against a *pinned* declared-edge snapshot so a
/// graph-conditioned predictor conditions on the consumer's declared relations.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ContextServeSource {
    /// Embedding-similarity context (`search`), the trained `context_k` wide.
    #[default]
    Ann,
    /// Declared-edge context: a bounded, target-anchored walk over a pinned
    /// edge snapshot.
    Edges(EdgeGather),
    /// The union of ANN and declared-edge context, pooled once.
    Hybrid {
        /// ANN neighbourhood size for the retrieval arm.
        ann_k: usize,
        /// The declared-edge gather for the edge arm.
        edges: EdgeGather,
        /// How the two candidate sets merge.
        merge: HybridMerge,
    },
}

impl ContextServeSource {
    /// The assembly-fact tag this serve source produces — the seam governance
    /// (E8) reads to attribute a coverage outcome's context.
    pub fn kind(&self) -> ContextSourceKind {
        match self {
            ContextServeSource::Ann => ContextSourceKind::Ann,
            ContextServeSource::Edges(_) => ContextSourceKind::Edges,
            ContextServeSource::Hybrid { .. } => ContextSourceKind::Hybrid,
        }
    }

    /// Build the per-target [`ContextSource`] the live context is assembled from.
    /// `Ann` uses the trained `context_k`; the edge/hybrid forms carry their own
    /// bounds.
    fn to_context_source(&self, context_k: usize) -> ContextSource {
        match self {
            ContextServeSource::Ann => ContextSource::Ann { k: context_k },
            ContextServeSource::Edges(gather) => ContextSource::Edges(gather.clone()),
            ContextServeSource::Hybrid {
                ann_k,
                edges,
                merge,
            } => ContextSource::Hybrid {
                ann_k: *ann_k,
                edges: edges.clone(),
                merge: *merge,
            },
        }
    }
}

/// Serving-time context options for a reloaded predictor: which live-context
/// source to assemble against (ANN or a pinned declared-edge snapshot) and an
/// optional serving-split predicate. [`Default`] is `(Ann, no split)` — exactly
/// the embedding-only behaviour a predictor without a declared edge source has.
#[derive(Debug, Clone, Default)]
pub struct ContextServeOptions {
    /// The live-context source.
    pub source: ContextServeSource,
    /// Optional serving-split predicate scoping the live context (e.g. a
    /// `split <> 'test'` corpus); `None` retrieves over the whole table.
    pub split: Option<String>,
}

/// A trained context predictor reloaded for inference: the rebuilt
/// [`AnyContextPredictor`] with its persisted weights, the served distribution
/// form, the source whose embedding table its live context is assembled from,
/// and the retrieval knobs (`context_k`, the serving context source + split).
/// Inference-only — it holds no optimizer and the forward never mutates its
/// [`VarMap`].
pub struct ServedContextPredictor {
    predictor: AnyContextPredictor,
    form: DistributionForm,
    source_id: String,
    table: ResultTableRecord,
    feature_dim: usize,
    context_k: usize,
    /// The source column carrying each context member's outcome `y` — the
    /// conditioning label an NP reads off the context, hydrated at serving time
    /// exactly as the episodic sampler hydrated it at training time.
    value_column: String,
    /// How the live context is assembled (ANN, or a pinned declared-edge
    /// snapshot) and the optional serving-split predicate.
    serve: ContextServeOptions,
    device: Device,
}

impl ServedContextPredictor {
    /// The served distribution form (Gaussian or the exact quantile levels).
    pub fn form(&self) -> &DistributionForm {
        &self.form
    }

    /// How this predictor's live context is assembled — the assembly *fact*
    /// (ANN / declared-edge / hybrid). The seam governance reads to attribute a
    /// served outcome's context and decide whether a marginal coverage claim over
    /// it is sound (E8). A fact, not an exchangeability judgment.
    pub fn source_kind(&self) -> ContextSourceKind {
        self.serve.source.kind()
    }

    /// The full serving context source (incl. any pinned edge snapshot + as-of).
    pub fn serve_source(&self) -> &ContextServeSource {
        &self.serve.source
    }
}

impl InferenceSession {
    /// Reload a trained context predictor for inference: read its catalog config,
    /// rebuild the [`AnyContextPredictor`] the config selects into a fresh
    /// [`VarMap`], and load the persisted safetensors weights into it. The
    /// returned [`ServedContextPredictor`] is inference-only — there is no
    /// optimizer, and a `predict` forward never writes to the loaded varmap.
    ///
    /// `source_id` is the source whose embedding table the *live* context is
    /// assembled from at serving time; `options` carries the live-context source
    /// (ANN, or a pinned declared-edge snapshot — [`ContextServeOptions`]) and an
    /// optional serving-split predicate (the predictor conditions on a serving
    /// corpus, with the target itself always excluded). The source need not equal
    /// the training source — a predictor meta-trained on one corpus serves a
    /// target's neighbourhood in another of the same shape (the PFN inductive
    /// property). [`ContextServeOptions::default`] is embedding-only ANN context.
    pub async fn load_context_predictor(
        self: &Arc<Self>,
        model_id: &str,
        source_id: &str,
        options: ContextServeOptions,
    ) -> Result<ServedContextPredictor> {
        let record = self.catalog().get_model(model_id).await?.ok_or_else(|| {
            JammiError::Catalog(format!("context predictor '{model_id}' not found"))
        })?;
        let config: serde_json::Value = record
            .config_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .ok_or_else(|| {
                JammiError::Inference(format!(
                    "context predictor '{model_id}' has no parseable config_json"
                ))
            })?;

        let form = PredictiveHead::form_from_config(config.get("head").ok_or_else(|| {
            JammiError::Inference(format!(
                "context predictor '{model_id}' config carries no head form"
            ))
        })?)?;

        let read_usize = |key: &str| -> Result<usize> {
            config
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| {
                    JammiError::Inference(format!(
                        "context predictor '{model_id}' config missing '{key}'"
                    ))
                })
        };
        let architecture = match config.get("architecture").and_then(|v| v.as_str()) {
            Some("Cnp") => ContextArchitecture::Cnp,
            Some("AttnCnp") => ContextArchitecture::AttnCnp,
            Some("Tnp") => ContextArchitecture::Tnp,
            other => {
                return Err(JammiError::Inference(format!(
                    "context predictor '{model_id}' has an unknown architecture {other:?}"
                )))
            }
        };
        let feature_dim = read_usize("feature_dim")?;
        let context_k = read_usize("context_k")?;
        let hidden_dim = read_usize("hidden_dim")?;
        let num_heads = read_usize("num_heads")?;
        let num_layers = read_usize("num_layers")?;
        let head_width = read_usize("head_width")?;
        let value_column = config
            .get("value_column")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                JammiError::Inference(format!(
                    "context predictor '{model_id}' config missing 'value_column'"
                ))
            })?
            .to_string();

        let table = self
            .catalog()
            .resolve_embedding_table(source_id, None)
            .await?;
        let serve_dim = table.dimensions.ok_or_else(|| {
            JammiError::Inference(format!(
                "serving source '{source_id}' embedding table carries no vector dimension"
            ))
        })? as usize;
        if serve_dim != feature_dim {
            return Err(JammiError::Inference(format!(
                "context predictor '{model_id}' was trained on feature_dim {feature_dim} but \
                 serving source '{source_id}' has dimension {serve_dim}"
            )));
        }

        let device = crate::model::backend::candle::select_device(self.device_config());

        // Build the predictor into a fresh varmap (which creates its variables),
        // then load the trained weights over them — the standard candle reload.
        // The loaded varmap is never handed to an optimizer, so inference cannot
        // mutate it.
        let mut varmap = VarMap::new();
        let cfg = ContextPredictorConfig {
            architecture,
            context_k,
            feature_dim,
            value_dim: 1,
            hidden_dim,
            num_heads,
            num_layers,
            head_width,
        };
        let predictor = {
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            AnyContextPredictor::new(&cfg, vb)
                .map_err(|e| JammiError::Inference(format!("rebuild context predictor: {e}")))?
        };
        // The recorded `artifact_path` is the object-store prefix the training
        // worker wrote the weights under. Fetch the bundle into a local cache
        // dir (a no-op copy for a `file://` root) and load `model.safetensors`
        // from there — so a predictor trained on one host reloads on another.
        let prefix = record.artifact_path.as_deref().ok_or_else(|| {
            JammiError::Inference(format!(
                "context predictor '{model_id}' has no artifact path"
            ))
        })?;
        let prefix_url = jammi_db::storage::StorageUrl::parse(prefix)?;
        let local = self.artifact_store().fetch_artifact(&prefix_url).await?;
        let weights_path = local.dir().join("model.safetensors");
        varmap
            .load(&weights_path)
            .map_err(|e| JammiError::Inference(format!("load predictor weights: {e}")))?;

        Ok(ServedContextPredictor {
            predictor,
            form,
            source_id: source_id.to_string(),
            table,
            feature_dim,
            context_k,
            value_column,
            serve: options,
            device,
        })
    }

    /// Predict a target's distribution by assembling its **live** context and
    /// running the predictor's in-context forward once — no gradient update.
    ///
    /// The target's own stored vector is read by key (an ordinary SQL-surface
    /// read), used as the retrieval query, and the S16 context is assembled with
    /// `exclude_self` on (the target never enters its own context) and the
    /// served predictor's serving split. The padded episode flows through one
    /// `forward` to the `[1, head_width]` head, and those floats run through the
    /// S18 [`DistributionAdapter`] for the served distribution — the same
    /// float→distribution transform the fine-tune regression path serves
    /// through.
    ///
    /// Inference-only: the predictor's [`VarMap`] is never handed to an
    /// optimizer and `forward` does not write to it, so the served weights are
    /// byte-identical before and after.
    ///
    /// This is the bare-distribution wrapper over
    /// [`predict_with_context_predictor_provenanced`](Self::predict_with_context_predictor_provenanced);
    /// reach for the provenanced form when the context member keys and the
    /// assembly `source` fact matter (the never-unattributed coverage contract).
    pub async fn predict_with_context_predictor(
        self: &Arc<Self>,
        served: &ServedContextPredictor,
        target_key: &str,
    ) -> Result<PredictedDistribution> {
        Ok(self
            .predict_with_context_predictor_provenanced(served, target_key)
            .await?
            .distribution)
    }

    /// Predict a target's distribution **with its context provenance**: the
    /// served distribution plus the assembly `source` fact (ANN / declared-edge /
    /// hybrid) and the context member keys.
    ///
    /// A graph-conditioned prediction is therefore never *unattributed*: the
    /// `source` fact and the neighbour keys ride out of the serving layer so
    /// governance can see how the context was built and decide whether a marginal
    /// coverage claim over it is sound — exactly the seam S16-G's coverage
    /// doctrine requires (the engine surfaces the fact; governance chooses the
    /// lever).
    pub async fn predict_with_context_predictor_provenanced(
        self: &Arc<Self>,
        served: &ServedContextPredictor,
        target_key: &str,
    ) -> Result<PredictionWithProvenance> {
        let target_x = self
            .read_vector_by_key(&served.table, target_key)
            .await?
            .ok_or_else(|| {
                JammiError::Inference(format!(
                    "target '{target_key}' has no stored vector in '{}'",
                    served.table.table_name
                ))
            })?;

        let mut request =
            ContextRequest::new(&served.source_id, target_x.clone(), served.context_k);
        // The live context is assembled from the served source: ANN retrieval,
        // or a pinned declared-edge snapshot, exactly as configured at load.
        request.source = served.serve.source.to_context_source(served.context_k);
        request.exclude_self = true;
        request.exclude_key = Some(target_key.to_string());
        request.split = served.serve.split.clone();
        request.aggregator = SetAggregator::Mean;
        // Hydrate the members' outcomes — the conditioning label an NP reads off
        // its context. Serving conditions on the same `(x, y)` members training
        // did, so the live predict hydrates the value column exactly as the
        // episodic sampler did.
        request.value_columns = vec![served.value_column.clone()];
        let rep = self.assemble_context(&request).await?;
        let source = rep.source;
        let context_keys = rep.context_keys.clone();
        let member_x = self
            .read_member_vectors(&served.table, &rep.context_keys)
            .await?;
        let member_y = extract_value_column(&rep.value_rows, &served.value_column)?;
        if member_x.len() != member_y.len() {
            return Err(JammiError::Inference(format!(
                "served context member x/y count mismatch for target '{target_key}': \
                 {} vectors vs {} labels",
                member_x.len(),
                member_y.len()
            )));
        }

        let context = TargetContext {
            target_x,
            // The served target's own outcome is what we are predicting; the head
            // is read off the target token / pooled context, not the target's own
            // label, so a serving target carries a placeholder `y` the forward
            // never consumes.
            target_y: 0.0,
            member_x,
            member_y,
        };
        let episode = pad_episode(
            std::slice::from_ref(&context),
            served.feature_dim,
            served.context_k,
            &served.device,
        )?;

        let head = served
            .predictor
            .forward(&episode)
            .map_err(|e| JammiError::Inference(format!("context predictor forward: {e}")))?;
        let distribution = distribution_from_head(&head, &served.form)?;
        Ok(PredictionWithProvenance {
            distribution,
            source,
            context_keys,
        })
    }
}

/// A served prediction with the provenance the coverage layer attributes it by:
/// the predictive distribution, the assembly `source` fact, and the context
/// member keys. The `source` fact and keys are what ride the uncertainty-channel
/// `context_ref` provenance — so a marginal claim over a graph-assembled context
/// is never unattributed (the S16-G coverage contract).
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionWithProvenance {
    /// The served predictive distribution (Gaussian or quantiles).
    pub distribution: PredictedDistribution,
    /// How the conditioning context was assembled (ANN / declared-edge / hybrid).
    pub source: ContextSourceKind,
    /// The context member keys, in retrieval order — the neighbour provenance.
    pub context_keys: Vec<String>,
}

/// Turn a `[1, head_width]` head tensor into a [`PredictedDistribution`] through
/// the S18 [`DistributionAdapter`] for the given form. One target, so the
/// adapter serves a single row; its `(mean, std)` or sorted `quantile_{level}`
/// columns are read back into the typed distribution.
fn distribution_from_head(head: &Tensor, form: &DistributionForm) -> Result<PredictedDistribution> {
    let flat = head
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| JammiError::Inference(format!("read predictor head: {e}")))?;

    let adapter = match form {
        DistributionForm::Gaussian => DistributionAdapter::gaussian(),
        DistributionForm::Quantile { levels } => DistributionAdapter::quantile(levels.clone())?,
    };
    let output = BackendOutput {
        float_outputs: vec![flat],
        string_outputs: vec![],
        row_status: vec![true],
        row_errors: vec![String::new()],
        shapes: vec![(1, head.dim(1).unwrap_or(0))],
    };
    let columns = adapter.adapt(&output, 1)?;

    use arrow::array::Float32Array;
    let col_f32 = |i: usize| -> Result<f32> {
        columns
            .get(i)
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
            .map(|a| a.value(0))
            .ok_or_else(|| {
                JammiError::Inference(format!("predictor head column {i} is not a served float"))
            })
    };

    match form {
        DistributionForm::Gaussian => Ok(PredictedDistribution::Gaussian {
            mean: col_f32(0)?,
            std: col_f32(1)?,
        }),
        DistributionForm::Quantile { levels } => {
            let mut pairs = Vec::with_capacity(levels.len());
            for (i, &level) in levels.iter().enumerate() {
                pairs.push((level, col_f32(i)?));
            }
            Ok(PredictedDistribution::Quantile { levels: pairs })
        }
    }
}

/// A conformal wrap over a served context predictor: the calibrated
/// [`ConformalModel`] plus the served predictor's distribution form, so a
/// `predict` output is turned into a coverage-guaranteed interval. The
/// calibration set is **held-out tasks** (never the training tasks) — an
/// amortized posterior is sharp but not automatically calibrated (the HIGH
/// calibration contract), so the S17 conformal wrap restores coverage over a
/// disjoint calibration split.
pub struct ConformalContextPredictor {
    conformal: ConformalModel,
    form: DistributionForm,
}

impl ConformalContextPredictor {
    /// The nominal miscoverage level `alpha` the wrap targets — the served
    /// interval covers at `>= 1 - alpha` under exchangeability.
    pub fn alpha(&self) -> f64 {
        self.conformal.alpha()
    }

    /// Wrap one served distribution into its conformal interval `[lower, upper]`.
    ///
    /// A Gaussian head wraps with absolute-residual conformal centred on the
    /// predictive mean; a quantile head wraps with CQR over its lower/upper
    /// served quantiles — the score family the calibration chose. `group` is the
    /// test target's cohort for a Mondrian (group-conditional) wrap — supplied by
    /// governance, `None` for a marginal wrap; the engine never derives it.
    pub fn interval(
        &self,
        prediction: &PredictedDistribution,
        group: Option<&str>,
    ) -> Result<(f64, f64)> {
        match &self.form {
            DistributionForm::Gaussian => {
                self.conformal
                    .predict_interval(prediction.point_estimate() as f64, 0.0, 0.0, group)
            }
            DistributionForm::Quantile { .. } => {
                let (lo, hi) = prediction.quantile_bounds().ok_or_else(|| {
                    JammiError::Inference(
                        "CQR conformal wrap requires a quantile-head prediction".into(),
                    )
                })?;
                self.conformal
                    .predict_interval(0.0, lo as f64, hi as f64, group)
            }
        }
    }
}

/// The conformal calibration lever a caller / governance supplies. The engine
/// **applies** the chosen lever; it never **chooses** one. `Marginal` is the
/// default and always serves; `Mondrian`/`Weighted` route to the group-
/// conditional / importance-weighted S17 constructors with caller-supplied
/// cohorts / weights (one per calibration point).
///
/// Graph-assembled context can break the exchangeability split-conformal
/// *marginal* coverage assumes; the levers are the honest repair — but *which*
/// cohort or weights, and *whether* to apply them, is governance's call (the
/// verbatim doctrine: "choosing the cohorts is governance, not a serving
/// output"). The engine surfaces the `source` fact and applies what it is told.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalLevers {
    /// Plain marginal split-conformal (the default; always serves).
    Marginal,
    /// Group-conditional (Mondrian): one cohort key per calibration point.
    Mondrian {
        /// Cohort key per calibration point, in calibration order.
        groups: Vec<String>,
    },
    /// Importance-weighted: one weight per calibration point (NAPS / Barber).
    Weighted {
        /// Non-negative weight per calibration point, in calibration order.
        weights: Vec<f64>,
    },
}

impl InferenceSession {
    /// Calibrate a conformal wrap for a served context predictor on a **held-out**
    /// calibration set — a set of `(target_key, observed_y)` whose tasks are
    /// disjoint from the training tasks.
    ///
    /// For each calibration target, `predict` assembles the live context and runs
    /// the in-context forward, and the conformal score is taken against the
    /// observed outcome: `|y - μ|` for a Gaussian head (absolute-residual), or
    /// `max(q_lo - y, y - q_hi)` for a quantile head (CQR). The finite-sample
    /// quantile of those scores at `alpha` is the wrap's threshold. Composes S17
    /// — it does not modify the conformal primitive, only feeds it.
    ///
    /// `levers` is the caller / governance choice of marginal (the default,
    /// which always serves), Mondrian (group-conditional), or weighted
    /// (importance-weighted) calibration — *applied* here, never *chosen* here.
    /// Graph-assembled context can break exchangeability; supplying a cohort or
    /// weights is the honest repair, but the choice is governance's, not the
    /// engine's (the verbatim conformal doctrine).
    pub async fn calibrate_context_predictor_conformal(
        self: &Arc<Self>,
        served: &ServedContextPredictor,
        calibration: &[(String, f64)],
        alpha: f64,
        levers: ConformalLevers,
    ) -> Result<ConformalContextPredictor> {
        if calibration.is_empty() {
            return Err(JammiError::Inference(
                "conformal calibration requires at least one held-out target".into(),
            ));
        }
        // A supplied lever carries one cohort / weight per calibration point;
        // a length mismatch is a typed error, never a silent misalignment.
        match &levers {
            ConformalLevers::Marginal => {}
            ConformalLevers::Mondrian { groups } if groups.len() == calibration.len() => {}
            ConformalLevers::Weighted { weights } if weights.len() == calibration.len() => {}
            ConformalLevers::Mondrian { groups } => {
                return Err(JammiError::Inference(format!(
                    "Mondrian conformal needs one cohort per calibration point: \
                     {} cohorts vs {} points",
                    groups.len(),
                    calibration.len()
                )))
            }
            ConformalLevers::Weighted { weights } => {
                return Err(JammiError::Inference(format!(
                    "weighted conformal needs one weight per calibration point: \
                     {} weights vs {} points",
                    weights.len(),
                    calibration.len()
                )))
            }
        }

        let mut predictions = Vec::with_capacity(calibration.len());
        let mut lower = Vec::with_capacity(calibration.len());
        let mut upper = Vec::with_capacity(calibration.len());
        let mut observed = Vec::with_capacity(calibration.len());
        for (key, y) in calibration {
            let dist = self.predict_with_context_predictor(served, key).await?;
            match &served.form {
                DistributionForm::Gaussian => predictions.push(dist.point_estimate() as f64),
                DistributionForm::Quantile { .. } => {
                    let (lo, hi) = dist.quantile_bounds().ok_or_else(|| {
                        JammiError::Inference(
                            "CQR calibration requires a quantile-head prediction".into(),
                        )
                    })?;
                    lower.push(lo as f64);
                    upper.push(hi as f64);
                }
            }
            observed.push(*y);
        }

        let score = match &served.form {
            DistributionForm::Gaussian => IntervalScore::AbsoluteResidual,
            DistributionForm::Quantile { .. } => IntervalScore::Cqr,
        };
        // The score family decides which arrays carry signal (point estimates for
        // absolute-residual, quantile bounds for CQR); the lever decides which
        // existing S17 constructor those feed. `Marginal` reproduces the plain
        // split-conformal interval exactly.
        let conformal = match &levers {
            ConformalLevers::Marginal => {
                ConformalModel::regression(&predictions, &lower, &upper, &observed, score, alpha)?
            }
            ConformalLevers::Mondrian { groups } => ConformalModel::regression_mondrian(
                &predictions,
                &lower,
                &upper,
                &observed,
                groups,
                score,
                alpha,
            )?,
            ConformalLevers::Weighted { weights } => ConformalModel::regression_weighted(
                &predictions,
                &lower,
                &upper,
                &observed,
                weights,
                score,
                alpha,
            )?,
        };

        Ok(ConformalContextPredictor {
            conformal,
            form: served.form.clone(),
        })
    }
}

/// Escape single quotes in a SQL string literal (`'` → `''`). The task value is
/// a column value of arbitrary content; the task/value/key column *names* are
/// the caller's own configured identifiers (quoted), not data.
fn escape_sql_literal(s: &str) -> String {
    s.replace('\'', "''")
}

/// In-place Fisher-Yates over a splitmix64 stream: deterministic given `seed`,
/// so the train/test **task** partition is reproducible across runs (the
/// determinism the meta-overfitting contract reports a stable gap against).
fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Collect every non-null value of a `Utf8` column into a `Vec<String>` in
/// batch/row order.
fn collect_string_column(batches: &[RecordBatch], column: &str) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(arr) = b
            .column_by_name(column)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        {
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.push(arr.value(i).to_string());
                }
            }
        }
    }
    out
}

/// Collect `(_key, _y)` pairs from a `(Utf8, Float64)` scan, dropping any row
/// whose key or outcome is null.
fn collect_key_value(batches: &[RecordBatch]) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for b in batches {
        let keys = b
            .column_by_name("_key")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let ys = b
            .column_by_name("_y")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
        if let (Some(keys), Some(ys)) = (keys, ys) {
            for i in 0..keys.len() {
                if !keys.is_null(i) && !ys.is_null(i) {
                    out.push((keys.value(i).to_string(), ys.value(i)));
                }
            }
        }
    }
    out
}

/// Extract the per-member scalar outcome `y` from the value-column batches
/// [`crate::pipeline::context_set`] hydrated, in retrieval order. The column is
/// `arrow_cast` to `Float64` so an integer or float outcome both resolve.
fn extract_value_column(value_rows: &[RecordBatch], value_column: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for batch in value_rows {
        let col = batch.column_by_name(value_column).ok_or_else(|| {
            JammiError::FineTune(format!(
                "context value-column '{value_column}' missing from hydrated rows"
            ))
        })?;
        let casted = arrow::compute::cast(col, &arrow::datatypes::DataType::Float64)
            .map_err(|e| JammiError::FineTune(format!("value-column cast to f64: {e}")))?;
        let floats = casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "context value-column '{value_column}' did not cast to Float64"
                ))
            })?;
        for i in 0..floats.len() {
            if floats.is_null(i) {
                return Err(JammiError::FineTune(format!(
                    "context value-column '{value_column}' carries a null outcome"
                )));
            }
            out.push(floats.value(i));
        }
    }
    Ok(out)
}

/// Reorder the `(_row_id, vector)` scan rows to match `context_keys` (retrieval
/// order). The IN-list scan returns rows in scan order; the attentive members
/// rely on the context-member order, so the vectors are reassembled per key.
fn order_vectors_by_keys(
    batches: &[RecordBatch],
    table: &str,
    context_keys: &[String],
) -> Result<Vec<Vec<f32>>> {
    use arrow::array::{FixedSizeListArray, Float32Array};

    let mut by_key: HashMap<String, Vec<f32>> = HashMap::new();
    for batch in batches {
        // `_row_id` reads back as `Utf8` or `Utf8View` depending on the parquet
        // reader's string-view setting; cast to `Utf8` so the downcast is one
        // type regardless of how the column was materialised.
        let id_col = batch
            .column_by_name("_row_id")
            .ok_or_else(|| JammiError::Schema {
                table: table.to_string(),
                column: "_row_id".into(),
                expected: "Utf8".into(),
                actual: "missing".into(),
            })?;
        let id_utf8 = arrow::compute::cast(id_col, &arrow::datatypes::DataType::Utf8)
            .map_err(|e| JammiError::FineTune(format!("_row_id cast to Utf8: {e}")))?;
        let ids = id_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| JammiError::Schema {
                table: table.to_string(),
                column: "_row_id".into(),
                expected: "Utf8".into(),
                actual: format!("{}", id_col.data_type()),
            })?;
        let vectors = batch
            .column_by_name("vector")
            .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
            .ok_or_else(|| JammiError::Schema {
                table: table.to_string(),
                column: "vector".into(),
                expected: "FixedSizeList<Float32>".into(),
                actual: "missing".into(),
            })?;
        for i in 0..ids.len() {
            let v = vectors.value(i);
            let floats =
                v.as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| JammiError::Schema {
                        table: table.to_string(),
                        column: "vector".into(),
                        expected: "FixedSizeList<Float32>".into(),
                        actual: format!("{}", v.data_type()),
                    })?;
            by_key.insert(
                ids.value(i).to_string(),
                (0..floats.len()).map(|j| floats.value(j)).collect(),
            );
        }
    }

    context_keys
        .iter()
        .map(|k| {
            by_key.remove(k).ok_or_else(|| {
                JammiError::FineTune(format!(
                    "context member '{k}' had no stored vector in '{table}'"
                ))
            })
        })
        .collect()
}

/// Stack a task's per-target contexts into a padded [`ContextEpisode`]: dense
/// `[B, k, feature_dim]` / `[B, k, 1]` tensors with a `[B, k]` presence mask that
/// is `1` at a real member and `0` at a pad slot. A target with fewer than `k`
/// surviving members (self-exclusion, split scoping, or a sparse neighbourhood)
/// leaves its trailing slots masked — never attended over, never pooled in.
fn pad_episode(
    contexts: &[TargetContext],
    feature_dim: usize,
    k: usize,
    device: &Device,
) -> Result<ContextEpisode> {
    let b = contexts.len();
    let mut target_flat = Vec::with_capacity(b * feature_dim);
    let mut cx_flat = vec![0f32; b * k * feature_dim];
    let mut cy_flat = vec![0f32; b * k];
    let mut presence = vec![0f32; b * k];

    for (i, ctx) in contexts.iter().enumerate() {
        if ctx.target_x.len() != feature_dim {
            return Err(JammiError::FineTune(format!(
                "target x width {} != embedding dim {feature_dim}",
                ctx.target_x.len()
            )));
        }
        target_flat.extend_from_slice(&ctx.target_x);
        for (j, member) in ctx.member_x.iter().enumerate().take(k) {
            if member.len() != feature_dim {
                return Err(JammiError::FineTune(format!(
                    "context member x width {} != embedding dim {feature_dim}",
                    member.len()
                )));
            }
            let base = (i * k + j) * feature_dim;
            cx_flat[base..base + feature_dim].copy_from_slice(member);
            cy_flat[i * k + j] = ctx.member_y[j] as f32;
            presence[i * k + j] = 1.0;
        }
    }

    let target_x = Tensor::from_vec(target_flat, (b, feature_dim), device)
        .map_err(|e| JammiError::FineTune(format!("target_x tensor: {e}")))?;
    let context_x = Tensor::from_vec(cx_flat, (b, k, feature_dim), device)
        .map_err(|e| JammiError::FineTune(format!("context_x tensor: {e}")))?;
    let context_y = Tensor::from_vec(cy_flat, (b, k, 1), device)
        .map_err(|e| JammiError::FineTune(format!("context_y tensor: {e}")))?;
    let presence = Tensor::from_vec(presence, (b, k), device)
        .map_err(|e| JammiError::FineTune(format!("presence tensor: {e}")))?;
    Ok(ContextEpisode {
        target_x,
        context_x,
        context_y,
        presence,
    })
}

/// Build the [`AnyContextPredictor`] the spec selects into `varmap`, with the
/// feature dim taken from the resolved embedding table and the head width from
/// the configured output.
fn build_predictor(
    spec: &ContextPredictorTrainConfig,
    feature_dim: usize,
    varmap: &VarMap,
    device: &Device,
) -> Result<AnyContextPredictor> {
    let cfg = ContextPredictorConfig {
        architecture: spec.architecture,
        context_k: spec.context_k,
        feature_dim,
        value_dim: 1,
        hidden_dim: spec.hidden_dim,
        num_heads: spec.num_heads,
        num_layers: spec.num_layers,
        head_width: spec.head.head_width(),
    };
    let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
    AnyContextPredictor::new(&cfg, vb)
        .map_err(|e| JammiError::FineTune(format!("build context predictor: {e}")))
}
