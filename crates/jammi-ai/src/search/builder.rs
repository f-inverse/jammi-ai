use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use jammi_engine::error::{JammiError, Result};
use jammi_engine::store::ResultStore;

use crate::evidence::provenance::add_provenance;
use crate::operator::ann_search_exec::AnnSearchExec;
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
