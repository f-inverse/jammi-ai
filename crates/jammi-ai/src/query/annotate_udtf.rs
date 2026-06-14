//! The `annotate` DataFusion table function — model inference as a SQL relation.
//!
//! Registered on the engine's `SessionContext`, `annotate(...)` lets a caller
//! run a model over the columns of a registered relation from inside SQL:
//!
//! ```sql
//! SELECT a._row_id, a.vector
//! FROM annotate('local:/models/bert', 'text_embedding', 'docs.public.papers',
//!               'id', 'abstract') AS a
//! ```
//!
//! This is the remote peer of the in-process [`crate::query::QueryBuilder::annotate`].
//! Both descend through [`InferenceSession::annotate_plan`] — the one
//! model-over-columns operator — so a compound retrieval+inference query runs
//! the same plan node whether it was built in-process or parsed from SQL over
//! the Flight SQL lane. Exposing inference as a table function (open,
//! caller-shaped composition over the data) rather than a typed RPC is the
//! Flight-SQL-lane choice from the typed-vs-Flight guideline: `search` is the
//! closed jammi-defined primitive; compound `join`/`filter`/`annotate` is
//! open composition.
//!
//! The function arguments are all string literals:
//!
//! | position | meaning |
//! |----------|---------|
//! | 0 | model id (`local:<path>`, an HF repo id, or a fine-tuned id) |
//! | 1 | task (catalog snake-case: `text_embedding`, `classification`, …) |
//! | 2 | relation to scan (`<source>.public.<table>`) |
//! | 3 | key column — carried through as the output `_row_id` |
//! | 4.. | one or more content columns the model reads |
//!
//! The output schema is the inference prefix (`_row_id`, `_source`, `_model`,
//! `_status`, `_error`, `_latency_ms`) followed by the task's columns. The
//! caller joins back to the source on `_row_id = <key column>`.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Weak};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::{plan_err, DataFusionError};
use datafusion::datasource::TableType;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;

use jammi_db::sql::{quote_ident, quote_relation};
use jammi_db::ModelTask;

use crate::inference::schema::build_output_schema;
use crate::model::ModelSource;
use crate::session::InferenceSession;

/// The `annotate` table function. Holds a **weak** handle to the engine
/// session so its [`TableProvider`]s can load models and build inference plans.
///
/// The handle is weak by necessity: the session owns the `SessionContext` the
/// function is registered on, so a strong back-reference would form a cycle the
/// session could never drop. The session always outlives any query it is
/// running, so the upgrade in `call` / `scan` succeeds for every live query; a
/// failed upgrade means the session is gone and is surfaced as an error rather
/// than a panic.
#[derive(Clone)]
pub struct AnnotateTableFunction {
    session: Weak<InferenceSession>,
}

impl fmt::Debug for AnnotateTableFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnotateTableFunction").finish()
    }
}

impl AnnotateTableFunction {
    /// The SQL name the function registers under.
    pub const NAME: &'static str = "annotate";

    pub fn new(session: Weak<InferenceSession>) -> Self {
        Self { session }
    }

    fn session(&self) -> datafusion::error::Result<Arc<InferenceSession>> {
        self.session.upgrade().ok_or_else(|| {
            DataFusionError::Execution("annotate: the engine session has been dropped".into())
        })
    }
}

/// Pull a string literal out of a table-function argument, or a planning error
/// that names the position. Table-function arguments are literals, so anything
/// non-literal (a column reference, an expression) is a caller mistake.
fn string_arg(args: &[Expr], idx: usize, what: &str) -> datafusion::error::Result<String> {
    match args.get(idx) {
        Some(Expr::Literal(ScalarValue::Utf8(Some(s)), _)) => Ok(s.clone()),
        Some(Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)) => Ok(s.clone()),
        _ => plan_err!("annotate: argument {idx} ({what}) must be a non-null string literal"),
    }
}

impl TableFunctionImpl for AnnotateTableFunction {
    fn call(&self, args: &[Expr]) -> datafusion::error::Result<Arc<dyn TableProvider>> {
        // annotate(model, task, relation, key_column, content_column [, ...])
        if args.len() < 5 {
            return plan_err!(
                "annotate(model, task, relation, key_column, content_column, ...) \
                 needs at least 5 arguments, got {}",
                args.len()
            );
        }
        let model = string_arg(args, 0, "model")?;
        let task_str = string_arg(args, 1, "task")?;
        let relation = string_arg(args, 2, "relation")?;
        let key_column = string_arg(args, 3, "key_column")?;
        let content_columns = (4..args.len())
            .map(|i| string_arg(args, i, "content_column"))
            .collect::<datafusion::error::Result<Vec<_>>>()?;

        let task = ModelTask::try_from_db_str(&task_str)
            .map_err(|e| DataFusionError::Plan(format!("annotate: invalid task: {e}")))?;
        let model_source = ModelSource::parse(&model);

        // The output schema is known at plan time but the embedding dimension
        // (the size of the `vector` FixedSizeList) is a property of the model,
        // so the schema resolves by loading the model. `call` is synchronous and
        // runs inside the engine's multi-thread tokio runtime (the Flight SQL
        // request handler and the embedded `block_on`); `block_in_place` hands
        // the load off without stalling the runtime, then drives the async load
        // to completion. The model is then warm in the cache for `scan`.
        let session = self.session()?;
        let model_source_for_dim = model_source.clone();
        // The model is loaded here both for the embedding dim and, for a
        // regression head, its persisted distribution form — the schema's
        // regression columns (Gaussian `mean`/`std` vs quantile level columns)
        // depend on the form, so the planned schema must read it rather than
        // assume Gaussian.
        let (embedding_dim, regression_form) = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let guard = session
                    .model_cache()
                    .get_or_load(&model_source_for_dim, task, None)
                    .await
                    .map_err(|e| {
                        DataFusionError::Plan(format!("annotate: load model '{model}': {e}"))
                    })?;
                let dim = guard.model.embedding_dim();
                let form = guard.model.regression_form().cloned();
                Ok::<_, DataFusionError>((dim, form))
            })
        })?;

        // Build the output schema from the prefix + the task adapter. The
        // input schema is not consulted by `build_output_schema`, so an empty
        // placeholder is sufficient.
        let placeholder = Arc::new(arrow::datatypes::Schema::empty());
        let schema = build_output_schema(
            &task,
            &placeholder,
            &key_column,
            embedding_dim,
            regression_form.as_ref(),
        )
        .map_err(|e| DataFusionError::Plan(format!("annotate: output schema: {e}")))?;

        Ok(Arc::new(AnnotateTable {
            session,
            model: model_source,
            task,
            relation,
            key_column,
            content_columns,
            schema,
        }))
    }
}

/// The relation produced by one `annotate(...)` call. Its `scan` builds
/// `[scan relation] -> InferenceExec` through the shared
/// [`InferenceSession::annotate_plan`] operator.
struct AnnotateTable {
    session: Arc<InferenceSession>,
    model: ModelSource,
    task: ModelTask,
    relation: String,
    key_column: String,
    content_columns: Vec<String>,
    schema: SchemaRef,
}

impl fmt::Debug for AnnotateTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnotateTable")
            .field("relation", &self.relation)
            .field("task", &self.task)
            .field("content_columns", &self.content_columns)
            .finish()
    }
}

#[async_trait]
impl TableProvider for AnnotateTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        // Inference is row-wise over the scanned relation; a WHERE over the
        // annotated output cannot be pushed below the model. DataFusion keeps
        // the filter above this node as a `FilterExec`.
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // Resolve the input relation through the session's SQL planner — the
        // same federated catalog (and the same per-request tenant scope on the
        // Flight SQL path) that any other query sees. Selecting the key +
        // content columns is what the model reads; nothing else is scanned.
        let mut columns = vec![quote_ident(&self.key_column)];
        for c in &self.content_columns {
            if c != &self.key_column {
                columns.push(quote_ident(c));
            }
        }
        // `self.relation` is the unquoted dotted reference passed as the 3rd
        // `annotate(...)` argument (`<source>.public.<table>`); each part is
        // quoted independently so a hyphenated or reserved source/table name
        // resolves verbatim rather than being re-parsed.
        let sql = format!(
            "SELECT {} FROM {}",
            columns.join(", "),
            quote_relation(&self.relation)
        );

        let session_state = state
            .as_any()
            .downcast_ref::<datafusion::execution::context::SessionState>()
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "annotate: session does not expose a SessionState for planning".into(),
                )
            })?;
        let logical = session_state.create_logical_plan(&sql).await?;
        let input = session_state.create_physical_plan(&logical).await?;

        let annotated = self
            .session
            .annotate_plan(
                input,
                &self.model,
                self.task,
                &self.content_columns,
                &self.key_column,
            )
            .await
            .map_err(|e| DataFusionError::Execution(format!("annotate: {e}")))?;

        // Honour the projection DataFusion requests. It may subset *or* reorder
        // the columns, so apply it whenever present rather than only when the
        // count differs (an identity projection is a harmless passthrough).
        match projection {
            Some(proj) => project_plan(annotated, proj),
            None => Ok(annotated),
        }
    }
}

/// Apply a column projection to a plan via `ProjectionExec` so the table
/// function honours `SELECT <subset or reordering>` over its output.
fn project_plan(
    plan: Arc<dyn ExecutionPlan>,
    projection: &[usize],
) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let exprs = projection
        .iter()
        .map(|&i| {
            let field = schema.field(i);
            let expr = datafusion::physical_expr::expressions::col(field.name(), schema.as_ref())?;
            Ok((
                expr as Arc<dyn datafusion::physical_expr::PhysicalExpr>,
                field.name().to_string(),
            ))
        })
        .collect::<datafusion::error::Result<Vec<_>>>()?;
    Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
}
