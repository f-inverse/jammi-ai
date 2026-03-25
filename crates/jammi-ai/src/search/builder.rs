use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::{JoinType, NullEquality};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use jammi_engine::error::{JammiError, Result};

use crate::evidence::provenance::add_provenance;
use crate::operator::ann_search_exec::AnnSearchExec;
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::session::InferenceSession;

/// Fluent API for constructing search queries.
///
/// Each method adds a node to a DataFusion execution plan.
/// `.run()` executes the plan and adds evidence provenance columns.
///
/// After the initial ANN search, results are automatically hydrated by
/// joining back to the source table — so all original columns (e.g.
/// `title`, `abstract`, `assignee_id`) are available for downstream
/// operations like `.join()`, `.annotate()`, `.filter()`, `.select()`.
pub struct SearchBuilder {
    session: Arc<InferenceSession>,
    plan: Arc<dyn ExecutionPlan>,
    channels: Vec<String>,
    annotated: bool,
}

impl SearchBuilder {
    /// Start a search over an embedding table.
    ///
    /// Creates an ANN search plan, then automatically hydrates results
    /// by joining back to the source table to include all original columns.
    pub(crate) async fn new(
        session: Arc<InferenceSession>,
        source_id: &str,
        query_vec: Vec<f32>,
        k: usize,
        embedding_table: Option<&str>,
    ) -> Result<Self> {
        let table = session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)?;

        let result_store = session.result_store();

        let ann = AnnSearchExec::new(
            table.clone(),
            query_vec,
            k,
            result_store,
            session.context().clone(),
        )?;

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(ann);

        // Hydration: join ANN results back to the source to get original columns.
        // ANN output is (_row_id Utf8, _source_id Utf8, similarity Float32).
        // The source key column may be a different type, so we cast it to Utf8.
        // We also cast all string columns to VARCHAR to avoid Utf8View/Utf8 mismatches
        // from the Parquet reader.
        if let Some(ref key_col) = table.key_column {
            let source_table_name = session.find_table_name(&table.source_id)?;
            // Build column list that casts string columns to VARCHAR for compatibility
            let source_cols = build_hydration_select(
                session.context(),
                &format!("{}.public.{source_table_name}", table.source_id),
                key_col,
            )
            .await?;
            let sql = format!(
                "SELECT {source_cols} FROM {}.public.{source_table_name}",
                table.source_id
            );
            let df = session
                .context()
                .sql(&sql)
                .await
                .map_err(|e| JammiError::Other(format!("Hydration scan: {e}")))?;
            let source_plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
                df.create_physical_plan()
                    .await
                    .map_err(|e| JammiError::Other(format!("Hydration plan: {e}")))?,
            ));

            let left_col = col("_row_id", plan.schema().as_ref())
                .map_err(|e| JammiError::Other(format!("Hydration join: {e}")))?;
            let right_col = col("_join_key", source_plan.schema().as_ref())
                .map_err(|e| JammiError::Other(format!("Hydration join: {e}")))?;

            let join = HashJoinExec::try_new(
                plan,
                source_plan,
                vec![(left_col, right_col)],
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
            )
            .map_err(|e| JammiError::Other(format!("Hydration join: {e}")))?;

            // Drop the _join_key column (redundant with _row_id)
            plan = drop_column(Arc::new(join), "_join_key")?;

            // Re-sort by similarity descending (join doesn't preserve order)
            let sim_col = col("similarity", plan.schema().as_ref())
                .map_err(|e| JammiError::Other(format!("Hydration sort: {e}")))?;
            let sort_expr = datafusion::physical_expr::PhysicalSortExpr {
                expr: sim_col,
                options: arrow::compute::SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            };
            if let Some(ordering) = datafusion::physical_expr::LexOrdering::new(vec![sort_expr]) {
                plan = Arc::new(datafusion::physical_plan::sorts::sort::SortExec::new(
                    ordering, plan,
                ));
            }
        }

        Ok(Self {
            session,
            plan,
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

    /// Select specific columns from the results.
    pub fn select(mut self, columns: &[String]) -> Result<Self> {
        let schema = self.plan.schema();
        let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = columns
            .iter()
            .map(|name| {
                let expr = col(name, schema.as_ref())
                    .map_err(|e| JammiError::Other(format!("Select column '{name}': {e}")))?;
                Ok((expr as Arc<dyn PhysicalExpr>, name.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = ProjectionExec::try_new(exprs, self.plan)
            .map_err(|e| JammiError::Other(format!("ProjectionExec: {e}")))?;
        self.plan = Arc::new(projection);
        Ok(self)
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

        let table_name = self.session.find_table_name(source)?;
        let sql = format!("SELECT * FROM {source}.public.{table_name}");
        let df = self
            .session
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::Other(format!("Join: {e}")))?;
        let right_plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
            df.create_physical_plan()
                .await
                .map_err(|e| JammiError::Other(format!("Join: {e}")))?,
        ));

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
    pub async fn annotate(mut self, model: &str, task: &str, columns: &[String]) -> Result<Self> {
        let model_source = crate::model::ModelSource::parse(model);

        let model_task: crate::model::ModelTask = task.parse()?;

        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, model_task, None)
            .await?;
        let embedding_dim = guard.model.embedding_dim();
        drop(guard);

        let inference = InferenceExecBuilder::new(
            self.plan,
            model_source,
            model_task,
            columns.to_vec(),
            "_row_id".to_string(),
            String::new(),
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

/// Build a SELECT column list for hydration that casts Utf8View columns to VARCHAR
/// and adds a `_join_key` column from the key column cast to VARCHAR.
async fn build_hydration_select(
    ctx: &datafusion::prelude::SessionContext,
    table_ref: &str,
    key_col: &str,
) -> Result<String> {
    let df = ctx
        .sql(&format!("SELECT * FROM {table_ref} LIMIT 0"))
        .await
        .map_err(|e| JammiError::Other(format!("Schema introspection: {e}")))?;
    let schema = df.schema();

    let mut cols = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        let cast = match field.data_type() {
            arrow::datatypes::DataType::Utf8View | arrow::datatypes::DataType::LargeUtf8 => {
                format!("arrow_cast(\"{name}\", 'Utf8') AS \"{name}\"")
            }
            _ => format!("\"{name}\""),
        };
        cols.push(cast);
    }
    cols.push(format!("arrow_cast(\"{key_col}\", 'Utf8') AS _join_key"));
    Ok(cols.join(", "))
}

/// Drop a single column from a plan via ProjectionExec.
fn drop_column(
    plan: Arc<dyn ExecutionPlan>,
    column_to_drop: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name() != column_to_drop)
        .map(|(_, f)| {
            let expr = col(f.name(), schema.as_ref())
                .map_err(|e| JammiError::Other(format!("drop_column: {e}")))?;
            Ok((expr as Arc<dyn PhysicalExpr>, f.name().to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let projection = ProjectionExec::try_new(exprs, plan)
        .map_err(|e| JammiError::Other(format!("drop_column projection: {e}")))?;
    Ok(Arc::new(projection))
}
