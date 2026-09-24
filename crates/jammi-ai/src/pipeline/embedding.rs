use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::{CacheOutcome, CachePolicy, ResultStore, ReusedArtifact, SinkKind};
use tracing::Instrument;

use crate::session::InferenceSession;
use jammi_datafusion::ModelSource;
use jammi_datafusion::ModelTask;
use jammi_datafusion::RowOrder;
use jammi_datafusion::{plan_inference, InferenceSpec};

/// The described model's identity for one embedding definition: the model
/// source, its embedding width, and the output-affecting environment the
/// definition hash folds (backend, precision, content digest, quantization,
/// device). Described once per producer call, never loaded — the base embed
/// and every refresh build their descriptor + environment from this one
/// place, and a plan placed elsewhere leaves this process holding no
/// weights.
pub(crate) struct EmbeddingDefinition {
    pub(crate) model_source: ModelSource,
    pub(crate) embedding_dim: usize,
    pub(crate) env: jammi_db::store::manifest::MaterializationEnv,
}

/// Describe `model_id` for `task` and build its [`EmbeddingDefinition`].
pub(crate) async fn embedding_definition(
    session: &InferenceSession,
    model_id: &str,
    task: ModelTask,
) -> Result<EmbeddingDefinition> {
    let model_source = ModelSource::parse(model_id);
    let description = session.model_cache().describe(&model_source, task).await?;
    let embedding_dim = description.embedding_dim();
    let env = jammi_db::store::manifest::MaterializationEnv::of_models(
        session.compute_device(),
        vec![description.identity().clone()],
    );
    Ok(EmbeddingDefinition {
        model_source,
        embedding_dim,
        env,
    })
}

/// Build the embedding plan: scan `source_id`'s catalog table for
/// `key_column` + `columns`, then the one inference plan
/// ([`plan_inference`]) over it — keyed order, `model_source`/`task`, the
/// `_content_hash` passthrough. [`EmbeddingPipeline::run`] executes this plan
/// in-process and a cluster submitter ships it, so both write the same bytes
/// by construction (one plan-building site).
///
/// `embedding_dim` and `model_source` are the caller's own (already
/// described via `embedding_definition`) rather than re-derived here, so
/// this function never touches the model.
pub async fn build_embedding_plan(
    session: &InferenceSession,
    source_id: &str,
    model_source: ModelSource,
    task: ModelTask,
    columns: &[String],
    key_column: &str,
    embedding_dim: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let table_name = session.find_table_name(source_id).await?;
    let query = session.build_source_query(source_id, &table_name, key_column, columns);

    let df = session
        .context()
        .sql(&query)
        .await
        .map_err(|e| JammiError::Inference(format!("Failed to scan source: {e}")))?;
    let input_plan = df
        .create_physical_plan()
        .await
        .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))?;
    // The source scan's `_content_hash` projection rides through to the sink
    // as the table's fifth column.
    let inference = &session.inner_config().inference;
    let spec = InferenceSpec {
        source: model_source,
        task,
        content_columns: columns.to_vec(),
        key_column: key_column.to_string(),
        source_id: source_id.to_string(),
        chunk: inference.chunk_budget()?,
        embedding_dim: Some(embedding_dim),
        regression_form: None,
        passthrough: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
        device_kind: session.required_device_kind(),
        partitions: inference.fan_out()?,
    };
    let plan = plan_inference(
        input_plan,
        RowOrder::Keyed {
            key_column: key_column.to_string(),
            tie_breakers: vec![jammi_db::store::schema::CONTENT_HASH_COLUMN.to_string()],
        },
        spec,
        session.inference_runtime(),
    )?;

    Ok(plan)
}

/// Orchestrates embedding generation: source scan → InferenceExec → the
/// result-table sink → index.
///
/// Modality-agnostic — works for both text (`ModelTask::TextEmbedding`) and
/// image (`ModelTask::ImageEmbedding`) by dispatching through InferenceExec.
pub struct EmbeddingPipeline<'a> {
    session: &'a InferenceSession,
    result_store: &'a ResultStore,
    task: ModelTask,
}

impl<'a> EmbeddingPipeline<'a> {
    pub fn new(
        session: &'a InferenceSession,
        result_store: &'a ResultStore,
        task: ModelTask,
    ) -> Self {
        Self {
            session,
            result_store,
            task,
        }
    }

    /// Run the embedding pipeline: scan source → run inference → persist to Parquet + index.
    ///
    /// `cache` opts into memoization. Embeddings anchor their source as
    /// [`AnchorKind::UnpinnedAtInstant`](jammi_db::store::manifest::AnchorKind::UnpinnedAtInstant)
    /// — a raw source has no version surface in open-core — so an `Use` request
    /// is honestly **always** a miss (`probe_cache` short-circuits any unpinned
    /// anchor): the cache is off here until sources expose a version surface. The
    /// returned [`CacheOutcome`] is therefore always `Computed`; the probe still
    /// runs so the surface is uniform and the honest off-ness is provable.
    pub async fn run(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        cache: CachePolicy,
        job_attempt: Option<jammi_db::catalog::result_repo::JobAttempt<'_>>,
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        // Describe the model: its embedding width and the output-affecting
        // environment the definition hash folds. Nothing is loaded here —
        // the plan's executing process materializes the weights.
        let EmbeddingDefinition {
            model_source,
            embedding_dim,
            env,
        } = embedding_definition(self.session, model_id, self.task)
            .instrument(tracing::debug_span!("embed.definition"))
            .await?;

        // The materialization contract is knowable here — the model is
        // described (so `embedding_dim` is fixed) and the source is named — so the cache
        // probe keys on the identical definition + anchors the funnel records at
        // finalize. The sole input is the raw source with no version surface →
        // `UnpinnedAtInstant`, so the probe is honestly always a miss.
        let canonical_model_id = model_source.to_string();
        let descriptor = jammi_db::store::manifest::ProducingDescriptor::Embedding {
            model_id: canonical_model_id.clone(),
            task: self.task,
            source_id: source_id.to_string(),
            columns: columns.to_vec(),
            key_column: key_column.to_string(),
            dimensions: embedding_dim,
        };
        let inputs = vec![jammi_db::store::manifest::InputAnchor::unpinned_at_instant(
            source_id,
            chrono::Utc::now().to_rfc3339(),
        )];

        if cache == CachePolicy::Use {
            let def_hash = jammi_db::store::manifest::MaterializationManifest::definition_of(
                &descriptor,
                &env,
            )
            .map_err(jammi_db::store::manifest_to_jammi)?;
            if let Some(reused) = self
                .result_store
                .probe_cache_record(&def_hash, &inputs)
                .await?
            {
                let outcome = CacheOutcome::Reused(ReusedArtifact::Table(reused.name()));
                return Ok((reused, outcome));
            }
        }

        // Create the lease-owned building table in the catalog. Every `?`
        // between here and `finish` unwinds through the handle's Drop (a
        // best-effort `building -> failed` CAS, no byte deletion).
        let col_list = columns.join(",");
        let mut building = self
            .result_store
            .create_table(
                source_id,
                self.task,
                jammi_db::catalog::result_repo::ResultTableKind::Model,
                None,
                &canonical_model_id,
                Some(embedding_dim as i32),
                Some(key_column),
                Some(&col_list),
                job_attempt,
            )
            .instrument(tracing::debug_span!("embed.create_table"))
            .await?;

        // Build the plan through the one plan-building site: in-process here and a Ballista
        // submitter both call `build_embedding_plan`, so both build byte-identical
        // plans by construction.
        let inference_exec = build_embedding_plan(
            self.session,
            source_id,
            model_source,
            self.task,
            columns,
            key_column,
            embedding_dim,
        )
        .instrument(tracing::debug_span!("embed.plan"))
        .await?;

        // Write through the sink, where the compute plane says: the ok rows
        // in the embedding schema, the segments built at the table's own
        // precision (`create_table` just stamped today's deployment default
        // on the row, so the same knobs apply here), a checkpoint every
        // `checkpoint_interval` batches.
        let embedding = &self.session.inner_config().embedding;
        let summary = self
            .result_store
            .write_result_table(
                &mut building,
                SinkKind::Embeddings {
                    dimensions: embedding_dim,
                    ann: embedding.ann,
                    segment_rows: embedding.index_segment_rows,
                    checkpoint_interval: embedding.checkpoint_interval,
                },
                inference_exec,
                self.session.context().task_ctx(),
            )
            .instrument(tracing::debug_span!("embed.write"))
            .await?;

        // Fail loud when there is nothing to embed. A systemic model failure (a
        // broken kernel / arch / dtype, any non-OOM `model.forward` error)
        // propagates from the runner as an `Err` out of the sink above, so it
        // never reaches here. What CAN reach here with zero written rows is a
        // source whose entire content column is empty/null: those rows fail
        // PRE-forward input validation and return `Ok(all-`_status = error`)`,
        // not an `Err`, so they cannot propagate. The sink drops error rows
        // and `finish` would then flip the catalog row to `ready` with
        // `row_count = 0` — a silently-empty table that searches to nothing,
        // with no signal. Detect all-input-invalid and error out. (A partial
        // failure still drops only the invalid rows.)
        if summary.input_rows > 0 && summary.rows == 0 {
            return Err(JammiError::Inference(
                "embedding generation produced no embeddings — every input row was empty \
                 or invalid (no valid content to embed); check the content column mapping"
                    .into(),
            ));
        }

        // Finish with the descriptor and anchors built at the top and the
        // environment the process that ran the plan reports (the cache probe
        // keyed on this process's prediction of it): renew the lease, write
        // the manifest sidecar, flip the catalog row `building -> ready` by
        // CAS, and register in DataFusion.
        let record = building
            .finish(
                self.session.context(),
                summary.rows as usize,
                jammi_db::store::manifest::Materialization::new(&descriptor, &summary.env, inputs),
            )
            .instrument(tracing::debug_span!("embed.finish"))
            .await?;
        Ok((record, CacheOutcome::Computed))
    }
}
