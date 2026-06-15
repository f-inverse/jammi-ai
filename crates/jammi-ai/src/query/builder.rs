use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{Field, Schema};
use datafusion::common::{JoinType, NullEquality};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::sql::source_relation;
use jammi_db::ChannelId;

use crate::evidence::{merge_channels, ChannelContribution};
use crate::operator::ann_search_exec::AnnSearchExec;
use crate::session::InferenceSession;

/// Fluent API for constructing vector-search-seeded compound queries.
///
/// Each method adds a node to a DataFusion execution plan.
/// `.run()` executes the plan and adds evidence provenance columns.
///
/// After the initial ANN search, results are automatically hydrated by
/// joining back to the source table — so all original columns (e.g.
/// `title`, `abstract`, `assignee_id`) are available for downstream
/// operations like `.join()`, `.annotate()`, `.filter()`, `.select()`.
///
/// This is the in-process compound surface. Its remote peer is the Flight SQL
/// lane, where the same `annotate` (model-over-columns) operation is exposed as
/// a DataFusion table function — both descend through
/// [`InferenceSession::annotate_plan`], so a compound query runs the same plan
/// node whether it was built in-process or parsed from remote SQL.
pub struct QueryBuilder {
    session: Arc<InferenceSession>,
    plan: Arc<dyn ExecutionPlan>,
    channels: Vec<ChannelId>,
    annotated: bool,
}

impl QueryBuilder {
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
            .resolve_embedding_table(source_id, embedding_table)
            .await?;

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
            let relation = source_relation(&table.source_id, &source_table_name);
            // Build column list that casts string columns to VARCHAR for compatibility
            let source_cols = build_hydration_select(session.context(), &relation, key_col).await?;
            let sql = format!("SELECT {source_cols} FROM {relation}");
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
            channels: vec![ChannelId::new("vector")?],
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
        let sql = format!("SELECT * FROM {}", source_relation(source, &table_name));
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

    /// Annotate query results by running model inference over selected columns.
    ///
    /// Delegates to [`InferenceSession::annotate_plan`] — the one
    /// model-over-columns operator shared with the Flight SQL `annotate` table
    /// function. The hydrated query rows carry `_row_id`, so the inference
    /// output keys back to them through it.
    pub async fn annotate(
        mut self,
        model: &str,
        task: crate::model::ModelTask,
        columns: &[String],
    ) -> Result<Self> {
        let model_source = crate::model::ModelSource::parse(model);
        self.plan = self
            .session
            .annotate_plan(self.plan, &model_source, task, columns, "_row_id")
            .await?;
        self.channels.push(ChannelId::new("inference")?);
        self.annotated = true;
        Ok(self)
    }

    /// Execute the plan and return results with evidence provenance.
    ///
    /// Output columns = source columns - (declared channel columns
    /// extracted from the source) + `retrieved_by` + `annotated_by` +
    /// per-channel declared columns in `(priority, ordinal)` order.
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

        let inference = ChannelId::new("inference")?;
        let retrieved: Vec<ChannelId> = self
            .channels
            .iter()
            .filter(|c| !(self.annotated && *c == &inference))
            .cloned()
            .collect();
        let annotated: Vec<ChannelId> = if self.annotated {
            vec![inference]
        } else {
            Vec::new()
        };

        // Pull declared channel columns out of each batch before
        // appending them via `merge_channels`. This avoids duplicating a
        // column (`similarity`) that would otherwise appear in both the
        // source and the suffix.
        let catalog = self.session.catalog();
        let mut per_batch_contribs: Vec<Vec<ChannelContribution>> =
            Vec::with_capacity(batches.len());
        let mut stripped: Vec<RecordBatch> = Vec::with_capacity(batches.len());
        for batch in &batches {
            let (rest, contribs) =
                extract_channel_contributions(batch, &self.channels, catalog).await?;
            per_batch_contribs.push(contribs);
            stripped.push(rest);
        }

        merge_channels(
            catalog,
            &stripped,
            &self.channels,
            &retrieved,
            &annotated,
            &per_batch_contribs,
        )
        .await
    }
}

/// Slice the channel-declared columns out of `batch` and return them as
/// per-channel contributions, alongside the batch with those columns
/// removed.
///
/// A channel contributes only if **all** of its declared columns are
/// present in the source batch under their declared names. If any are
/// missing, the channel produces no contribution and its declared
/// columns become all-null in the merged output. Dtype mismatches are
/// not coerced here; `merge_channels`'s validator surfaces them as a
/// typed `ChannelAssembly` error so callers see the real mismatch.
async fn extract_channel_contributions(
    batch: &RecordBatch,
    participating: &[ChannelId],
    catalog: &Catalog,
) -> Result<(RecordBatch, Vec<ChannelContribution>)> {
    let mut contributions: Vec<ChannelContribution> = Vec::new();
    let mut to_remove: HashSet<usize> = HashSet::new();

    for id in participating {
        let spec = catalog.channels().get(id).await?.ok_or_else(|| {
            JammiError::ChannelAssembly(format!("channel '{id}': not registered"))
        })?;
        let positions: Option<Vec<usize>> = spec
            .columns
            .iter()
            .map(|c| batch.schema().index_of(&c.name).ok())
            .collect();
        if let Some(positions) = positions {
            let columns: Vec<ArrayRef> = positions
                .iter()
                .map(|&i| Arc::clone(batch.column(i)))
                .collect();
            for i in positions {
                to_remove.insert(i);
            }
            contributions.push(ChannelContribution {
                channel: id.clone(),
                columns,
            });
        }
    }

    let new_fields: Vec<Arc<Field>> = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_remove.contains(i))
        .map(|(_, f)| Arc::clone(f))
        .collect();
    let new_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .enumerate()
        .filter(|(i, _)| !to_remove.contains(i))
        .map(|(_, c)| Arc::clone(c))
        .collect();
    let new_schema = Arc::new(Schema::new(
        new_fields.iter().map(|f| (**f).clone()).collect::<Vec<_>>(),
    ));
    let stripped = RecordBatch::try_new(new_schema, new_columns)
        .map_err(|e| JammiError::ChannelAssembly(format!("extract_channel_contributions: {e}")))?;

    Ok((stripped, contributions))
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
