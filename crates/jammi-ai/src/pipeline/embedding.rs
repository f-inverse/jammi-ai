use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::store::{CacheOutcome, CachePolicy, ResultStore};

use crate::model::{ModelSource, ModelTask};
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::pipeline::result_sink::ResultSink;
use crate::session::InferenceSession;

/// Orchestrates embedding generation: source scan → InferenceExec → ResultSink → index.
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
    ) -> Result<(ResultTableRecord, CacheOutcome)> {
        let model_source = ModelSource::parse(model_id);

        // Pre-load model to get embedding dimensions
        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, self.task, None)
            .await?;
        let embedding_dim = guard
            .model
            .embedding_dim()
            .ok_or_else(|| JammiError::Inference("Model does not support embeddings".into()))?;
        let backend_kind = guard.model.backend_kind();
        let compute_precision = guard.model.compute_precision();
        let content_digest = guard.model.content_digest()?;
        let quantization = guard.model.quantization();
        drop(guard);

        // The materialization contract is knowable here — the model is loaded
        // (so `embedding_dim` is fixed) and the source is named — so the cache
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
        let env = jammi_db::store::manifest::MaterializationEnv::new(
            self.session.compute_device(),
            vec![jammi_db::store::manifest::ModelIdentity {
                model_id: canonical_model_id.clone(),
                backend: backend_kind.to_string(),
                compute_precision,
                content_digest,
                quantization,
            }],
        );
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
                let table = reused.table_name.clone();
                return Ok((reused, CacheOutcome::Reused { table }));
            }
        }

        // Create result table in catalog
        let col_list = columns.join(",");
        let table_info = self
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
            )
            .await?;

        // Build scan plan over source
        let table_name = self.session.find_table_name(source_id)?;
        let query = self
            .session
            .build_source_query(source_id, &table_name, key_column, columns);

        let df = self
            .session
            .context()
            .sql(&query)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to scan source: {e}")))?;
        let input_plan = df
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))?;

        // Create InferenceExec
        let inference_exec = InferenceExecBuilder::new(
            input_plan,
            model_source,
            self.task,
            columns.to_vec(),
            key_column.to_string(),
            source_id.to_string(),
            Arc::clone(self.session.model_cache()),
        )
        .batch_size(self.session.inner_config().inference.batch_size)
        .observer(self.session.observer().clone())
        .embedding_dim(Some(embedding_dim))
        .build()?;

        // Create ResultSink
        let embedding_schema = jammi_db::store::schema::embedding_table_schema(embedding_dim);
        let writer = self
            .result_store
            .open_writer(&table_info.parquet_url, embedding_schema)
            .await?;
        // Fresh creation — `create_table` above just stamped this table's
        // catalog row with today's deployment default, so the same default
        // applies here (unlike a rebuild, there is no pre-existing catalog
        // promise to honour).
        let ann_config = &self.session.inner_config().embedding.ann;
        let sidecar = SidecarIndex::new(embedding_dim, ann_config, ann_config.storage_precision)?;
        let checkpoint_interval = self.session.inner_config().embedding.checkpoint_interval;
        let mut sink = ResultSink::for_embeddings(
            writer,
            sidecar,
            self.session.catalog(),
            table_info.table_name.clone(),
            checkpoint_interval,
        );

        // Execute and stream results through sink
        let task_ctx = self.session.context().task_ctx();
        let stream = inference_exec
            .execute(0, task_ctx)
            .map_err(|e| JammiError::Inference(format!("InferenceExec failed: {e}")))?;

        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to collect results: {e}")))?;

        // Fail loud when there is nothing to embed. A systemic model failure (a
        // broken kernel / arch / dtype — #277/#319/#326, any non-OOM
        // `model.forward` error) now propagates from the runner as an `Err` out
        // of `collect` above, so it never reaches here. What CAN reach here with
        // zero successful rows is a source whose entire content column is
        // empty/null: those rows fail PRE-forward input validation and return
        // `Ok(all-`_status = error`)`, not an `Err`, so they cannot propagate.
        // The sink drops error rows and `finalize` would then flip the catalog
        // row to `ready` with `row_count = 0` — a silently-empty table that
        // searches to nothing, with no signal. Detect all-input-invalid and
        // error out. (A partial failure still drops only the invalid rows.)
        {
            let mut total = 0usize;
            let mut ok = 0usize;
            for batch in &batches {
                let s_idx = batch.schema().index_of("_status").map_err(|e| {
                    JammiError::Inference(format!("inference output missing _status: {e}"))
                })?;
                let status = batch
                    .column(s_idx)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| JammiError::Inference("_status is not Utf8".into()))?;
                for i in 0..batch.num_rows() {
                    total += 1;
                    if status.value(i) == "ok" {
                        ok += 1;
                    }
                }
            }
            if total > 0 && ok == 0 {
                return Err(JammiError::Inference(
                    "embedding generation produced no embeddings — every input row was empty \
                     or invalid (no valid content to embed); check the content column mapping"
                        .into(),
                ));
            }
        }

        for batch in &batches {
            sink.write_batch(batch).await?;
        }

        let (row_count, index) = sink.finalize().await?;

        // Persist the built index as this table's first ANN segment (segment 0).
        // The `building` catalog row carries the table's persisted precision,
        // which `append_segment` checks the built index against (B4).
        if let Some(ref idx) = index {
            let building = self
                .session
                .catalog()
                .get_result_table(&table_info.table_name)
                .await?
                .ok_or_else(|| {
                    JammiError::Catalog(format!(
                        "Result table '{}' not found while persisting its index segment",
                        table_info.table_name
                    ))
                })?;
            self.result_store.append_segment(&building, idx).await?;
        }

        // Finalize with the contract built at the top (the same definition +
        // anchors the cache probe keyed on), write the manifest sidecar, register
        // in DataFusion, and flip the catalog row `building -> ready`.
        self.result_store
            .finalize_with_manifest(
                self.session.context(),
                &table_info.table_name,
                &table_info.parquet_url,
                row_count,
                jammi_db::store::manifest::Materialization::new(&descriptor, &env, inputs),
            )
            .await?;

        // Return the updated record
        let record = self
            .session
            .catalog()
            .get_result_table(&table_info.table_name)
            .await?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "Result table '{}' not found after finalization",
                    table_info.table_name
                ))
            })?;
        Ok((record, CacheOutcome::Computed))
    }
}
