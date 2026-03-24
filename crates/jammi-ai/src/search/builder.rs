use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::{JoinType, NullEquality};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use jammi_engine::error::{JammiError, Result};
use jammi_engine::store::ResultStore;

use crate::evidence::provenance::add_provenance;
use crate::operator::ann_search_exec::AnnSearchExec;
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::session::InferenceSession;

/// Fluent API for constructing search queries.
///
/// Each method adds a node to a DataFusion execution plan.
/// `.run()` executes the plan and adds evidence provenance columns.
///
/// ```text
/// session.search("patents", query_vec, 10).await?
///     .sort("similarity", true)?
///     .limit(5)
///     .run().await?
/// ```
pub struct SearchBuilder {
    session: Arc<InferenceSession>,
    plan: Arc<dyn ExecutionPlan>,
    channels: Vec<String>,
    annotated: bool,
}

impl SearchBuilder {
    /// Start a search over an embedding table.
    pub(crate) async fn new(
        session: Arc<InferenceSession>,
        source_id: &str,
        query_vec: Vec<f32>,
        k: usize,
        _embedding_table: Option<&str>,
    ) -> Result<Self> {
        let table = session
            .catalog()
            .resolve_embedding_table(source_id, _embedding_table)?;

        let result_store = Arc::new(ResultStore::new(
            session.inner_config().artifact_dir.as_path(),
            Arc::new(jammi_engine::catalog::Catalog::open(
                &session.inner_config().artifact_dir,
            )?),
        )?);

        let ann = AnnSearchExec::new(table, query_vec, k, result_store, session.context().clone())?;

        Ok(Self {
            session,
            plan: Arc::new(ann),
            channels: vec!["vector".into()],
            annotated: false,
        })
    }

    /// Filter results with a SQL predicate.
    pub fn filter(mut self, predicate: &str) -> Result<Self> {
        let arrow_schema = self.plan.schema();
        let df_schema = datafusion::common::DFSchema::try_from(arrow_schema.as_ref().clone())
            .map_err(|e| JammiError::Other(format!("DFSchema conversion: {e}")))?;
        let expr = datafusion::prelude::SessionContext::new()
            .parse_sql_expr(predicate, &df_schema)
            .map_err(|e| JammiError::Other(format!("Filter parse: {e}")))?;
        let execution_props = datafusion::execution::context::ExecutionProps::new();
        let physical_expr =
            datafusion::physical_expr::create_physical_expr(&expr, &df_schema, &execution_props)
                .map_err(|e| JammiError::Other(format!("Filter physical expr: {e}")))?;
        let filter =
            datafusion::physical_plan::filter::FilterExec::try_new(physical_expr, self.plan)
                .map_err(|e| JammiError::Other(format!("FilterExec: {e}")))?;
        self.plan = Arc::new(filter);
        Ok(self)
    }

    /// Sort results by a column.
    pub fn sort(mut self, column: &str, descending: bool) -> Result<Self> {
        let col_expr =
            datafusion::physical_expr::expressions::col(column, self.plan.schema().as_ref())
                .map_err(|e| JammiError::Other(format!("Sort column: {e}")))?;
        let sort_expr = datafusion::physical_expr::PhysicalSortExpr {
            expr: col_expr,
            options: arrow::compute::SortOptions {
                descending,
                nulls_first: false,
            },
        };
        let ordering = datafusion::physical_expr::LexOrdering::new(vec![sort_expr])
            .ok_or_else(|| JammiError::Other("Empty sort ordering".into()))?;
        let sort = datafusion::physical_plan::sorts::sort::SortExec::new(ordering, self.plan);
        self.plan = Arc::new(sort);
        Ok(self)
    }

    /// Limit the number of results.
    pub fn limit(mut self, n: usize) -> Self {
        let limit = datafusion::physical_plan::limit::GlobalLimitExec::new(self.plan, 0, Some(n));
        self.plan = Arc::new(limit);
        self
    }

    /// Join search results with another registered source.
    ///
    /// `on` is `"left_col=right_col"`. `how` is `"inner"` or `"left"` (default).
    pub async fn join(mut self, source: &str, on: &str, how: Option<&str>) -> Result<Self> {
        let (left_col_name, right_col_name) = on
            .split_once('=')
            .ok_or_else(|| JammiError::Other(format!("Join: 'on' must contain '=', got '{on}'")))?;

        let join_type = match how {
            Some("inner") => JoinType::Inner,
            _ => JoinType::Left,
        };

        // Scan the join source table
        let table_name = self.session.find_table_name(source)?;
        let sql = format!("SELECT * FROM {source}.public.{table_name}");
        let df = self
            .session
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::Other(format!("Join: {e}")))?;
        let right_plan = df
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Other(format!("Join: {e}")))?;

        // Build column expressions for the join keys
        let left_col = col(left_col_name, self.plan.schema().as_ref())
            .map_err(|e| JammiError::Other(format!("Join: {e}")))?;
        let right_col = col(right_col_name, right_plan.schema().as_ref())
            .map_err(|e| JammiError::Other(format!("Join: {e}")))?;

        let join = HashJoinExec::try_new(
            self.plan,
            right_plan,
            vec![(left_col, right_col)],
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
        )
        .map_err(|e| JammiError::Other(format!("Join: {e}")))?;

        self.plan = Arc::new(join);
        Ok(self)
    }

    /// Annotate search results by running model inference over selected columns.
    ///
    /// This inserts an `InferenceExec` node into the execution plan so that
    /// each result row is enriched with model outputs (e.g. embeddings,
    /// classifications) before final collection.
    pub async fn annotate(mut self, model: &str, task: &str, columns: &[String]) -> Result<Self> {
        // Parse model source from the model string
        let model_source = if let Some(path) = model.strip_prefix("local:") {
            crate::model::ModelSource::local(std::path::PathBuf::from(path))
        } else {
            crate::model::ModelSource::hf(model)
        };

        // Parse the task string to ModelTask enum
        let model_task = match task.to_lowercase().as_str() {
            "embedding" => crate::model::ModelTask::Embedding,
            "classification" => crate::model::ModelTask::Classification,
            "summarization" => crate::model::ModelTask::Summarization,
            "ner" => crate::model::ModelTask::Ner,
            "text_generation" => crate::model::ModelTask::TextGeneration,
            other => return Err(JammiError::Other(format!("Unknown task: {other}"))),
        };

        // Pre-load model to get embedding dimensions
        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, model_task, None)
            .await?;
        let embedding_dim = guard.model.embedding_dim();
        drop(guard);

        // Build InferenceExec using the builder
        let inference = InferenceExecBuilder::new(
            self.plan,
            model_source,
            model_task,
            columns.to_vec(),
            "_row_id".to_string(),
            String::new(), // source_id not needed for annotation
            std::sync::Arc::clone(self.session.model_cache()),
        )
        .batch_size(self.session.inner_config().inference.batch_size)
        .observer(self.session.observer().clone())
        .embedding_dim(embedding_dim)
        .build()?;

        self.plan = Arc::new(inference);
        self.channels.push("inference".into());
        self.annotated = true;
        Ok(self)
    }

    /// Execute the plan and return results with evidence provenance.
    pub async fn run(self) -> Result<Vec<RecordBatch>> {
        let task_ctx = self.session.context().task_ctx();
        let stream = self
            .plan
            .execute(0, task_ctx)
            .map_err(|e| JammiError::Other(format!("Search execute: {e}")))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| JammiError::Other(format!("Search collect: {e}")))?;

        add_provenance(&batches, &self.channels, self.annotated)
    }
}
