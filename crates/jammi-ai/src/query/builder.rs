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
use jammi_db::index::{validate_query, QuerySource};
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
    /// `source` is the query's provenance and decides whose fault a bad one
    /// is: a caller's vector fails as a schema-class error (`InvalidArgument`
    /// on the wire); a vector read back from storage (query-by-example) fails
    /// as a corrupt artifact named by its table. This is the EARLIEST point a
    /// query can be refused — the first place any width is known — so a
    /// caller fault never reaches placement, let alone the failure ladder.
    /// The catalog width is applied to a CALLER's vector when recorded; a
    /// STORED vector gets the finiteness check only (it came from the same
    /// column as the corpus, so a catalog/data drift must not refuse a valid
    /// self-query). Every entry downstream enforces the width it knows.
    pub(crate) async fn new(
        session: Arc<InferenceSession>,
        source_id: &str,
        query_vec: Vec<f32>,
        k: usize,
        embedding_table: Option<&str>,
        oversample: Option<usize>,
        source: QuerySource,
    ) -> Result<Self> {
        let table = session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;

        let width = match &source {
            QuerySource::Caller => table.dimensions().map(std::num::NonZeroUsize::get),
            QuerySource::Stored { .. } => None,
        };
        let query_vec = validate_query(query_vec, width, source)?;

        let result_store = session.result_store();

        let ann = AnnSearchExec::new(
            table.clone(),
            query_vec,
            k,
            oversample,
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
            .map_err(|e| plan_error("Search execute", e))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| plan_error("Search collect", e))?;

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

/// The engine error a plan raised, recovered across the DataFusion boundary.
///
/// A plan leaf (the ANN search leaf) boxes its `JammiError` into
/// [`DataFusionError::External`] — possibly under a `Context` wrapper, and
/// handed through the stream machinery as a `Shared(Arc<_>)` — so the typed
/// variant is unwrapped here rather than flattened into a string: a typed
/// refusal such as [`JammiError::Unavailable`] keeps its variant (and its
/// gRPC code) across `Search`. The wrappers are walked only to FIND a typed
/// engine error; when none is found the ORIGINAL error's own `Display` text
/// — its context description and `caused by` chain included — is kept
/// exactly, under the `"{stage}: "` prefix it always had.
fn plan_error(stage: &str, e: datafusion::error::DataFusionError) -> JammiError {
    match extract_engine_error(e) {
        Ok(engine) => engine,
        Err(original) => JammiError::Other(format!("{stage}: {original}")),
    }
}

/// Take the typed [`JammiError`] out of `e` if one is boxed anywhere under
/// it; otherwise hand `e` back UNCHANGED (rebuilt with the same wrappers) so
/// its `Display` text is exactly what it was.
fn extract_engine_error(
    e: datafusion::error::DataFusionError,
) -> std::result::Result<JammiError, datafusion::error::DataFusionError> {
    use datafusion::error::DataFusionError;
    match e {
        DataFusionError::External(boxed) => match boxed.downcast::<JammiError>() {
            Ok(engine) => Ok(*engine),
            // The stream machinery can re-wrap a DataFusion error as
            // `External(Box<DataFusionError>)`: walk through it, rebuilding
            // the wrapper on a miss so the text is unchanged.
            Err(other) => match other.downcast::<DataFusionError>() {
                Ok(nested) => extract_engine_error(*nested)
                    .map_err(|inner| DataFusionError::External(Box::new(inner))),
                Err(other) => Err(DataFusionError::External(other)),
            },
        },
        DataFusionError::Context(description, inner) => match extract_engine_error(*inner) {
            Ok(engine) => Ok(engine),
            Err(inner) => Err(DataFusionError::Context(description, Box::new(inner))),
        },
        DataFusionError::Shared(arc) => match Arc::try_unwrap(arc) {
            Ok(inner) => extract_engine_error(inner)
                .map_err(|inner| DataFusionError::Shared(Arc::new(inner))),
            // Still shared elsewhere (the hydration join keeps its build
            // side's error alive): `JammiError` is not `Clone`, so the typed
            // variants the leaf raises are rebuilt by reference; everything
            // else keeps the original text.
            Err(arc) => match engine_error_by_ref(&arc) {
                Some(engine) => Ok(engine),
                None => Err(DataFusionError::Shared(arc)),
            },
        },
        other => Err(other),
    }
}

/// FIND a typed engine error anywhere under `e` (through `Context` / `Shared`
/// / nested `External` wrappers) and rebuild it BY REFERENCE — the arm for a
/// `Shared(Arc<_>)` that is still shared elsewhere (the hydration join keeps
/// its build side's error alive), where the box cannot be moved out.
/// `JammiError` is not `Clone`, so this rebuilds field by field the variants
/// the search leaf raises: a `Schema` refusal, a corrupt-artifact
/// `IncompatibleFormat`, a peer-ladder `Unavailable`, and the string-carrying
/// engine faults. Anything else keeps the original text. A finder only: it
/// never produces text, so a `Context` description is never dropped by it.
fn engine_error_by_ref(e: &datafusion::error::DataFusionError) -> Option<JammiError> {
    use datafusion::error::DataFusionError;
    match e {
        DataFusionError::External(boxed) => match boxed.downcast_ref::<JammiError>() {
            Some(engine) => rebuild_engine_error(engine),
            None => boxed
                .downcast_ref::<DataFusionError>()
                .and_then(engine_error_by_ref),
        },
        DataFusionError::Context(_, inner) => engine_error_by_ref(inner),
        DataFusionError::Shared(arc) => engine_error_by_ref(arc),
        _ => None,
    }
}

/// Field-by-field rebuild of the engine variants a plan leaf raises.
fn rebuild_engine_error(e: &JammiError) -> Option<JammiError> {
    Some(match e {
        JammiError::Unavailable { resource, reason } => JammiError::Unavailable {
            resource: resource.clone(),
            reason: reason.clone(),
        },
        JammiError::Schema {
            table,
            column,
            expected,
            actual,
        } => JammiError::Schema {
            table: table.clone(),
            column: column.clone(),
            expected: expected.clone(),
            actual: actual.clone(),
        },
        JammiError::IncompatibleFormat {
            artifact,
            found,
            supported,
        } => JammiError::IncompatibleFormat {
            artifact: artifact.clone(),
            found: found.clone(),
            supported: supported.clone(),
        },
        JammiError::Catalog(m) => JammiError::Catalog(m.clone()),
        JammiError::Config(m) => JammiError::Config(m.clone()),
        JammiError::Inference(m) => JammiError::Inference(m.clone()),
        JammiError::Other(m) => JammiError::Other(m.clone()),
        _ => return None,
    })
}

#[cfg(test)]
mod plan_error_tests {
    use super::*;
    use datafusion::error::DataFusionError;

    fn ctx_err() -> DataFusionError {
        DataFusionError::Plan("bad plan".into()).context("while planning the scan")
    }

    /// A non-engine `Context` error keeps its own `Display` text exactly —
    /// the description AND the `caused by` chain — under the stage prefix.
    #[test]
    fn non_engine_context_error_keeps_its_full_display_text() {
        let expected = format!("Search execute: {}", ctx_err());
        assert!(expected.contains("while planning the scan"));
        assert!(expected.contains("caused by"));
        let got = plan_error("Search execute", ctx_err());
        assert_eq!(got.to_string(), expected);
        assert!(matches!(got, JammiError::Other(_)));
    }

    /// The same shape for a `Shared` non-engine error: text kept verbatim.
    #[test]
    fn non_engine_shared_context_error_keeps_its_full_display_text() {
        let shared = DataFusionError::Shared(Arc::new(ctx_err()));
        let expected = format!("Search collect: {shared}");
        let got = plan_error("Search collect", shared);
        assert_eq!(got.to_string(), expected);
    }

    /// The stream machinery's `External(Box<DataFusionError>)` re-wrap is
    /// walked through too — the exact-path sink's typed error must survive
    /// `Search collect` as itself, not as `Other("External error: …")`.
    #[test]
    fn engine_error_under_a_nested_external_is_recovered_typed() {
        let inner = DataFusionError::External(Box::new(JammiError::IncompatibleFormat {
            artifact: "t.vector".into(),
            found: "row 'x' yields a non-finite distance".into(),
            supported: "finite f32 components".into(),
        }));
        let outer = DataFusionError::External(Box::new(inner));
        assert!(matches!(
            plan_error("Search collect", outer),
            JammiError::IncompatibleFormat { .. }
        ));
        // A non-engine nested External keeps its text verbatim.
        let plain = DataFusionError::External(Box::new(DataFusionError::Plan("bad".into())));
        let expected = format!("Search collect: {plain}");
        assert_eq!(plan_error("Search collect", plain).to_string(), expected);
    }

    /// A `Shared(Arc<_>)` that is STILL SHARED (the hydration join's build
    /// side) cannot be moved out of; the leaf's typed variants are rebuilt by
    /// reference — the corrupt-artifact and schema classes included, not only
    /// `Unavailable`.
    #[test]
    fn still_shared_engine_errors_are_rebuilt_by_reference() {
        for engine in [
            JammiError::IncompatibleFormat {
                artifact: "t.vector".into(),
                found: "row 'x' yields a non-finite distance".into(),
                supported: "finite f32 components".into(),
            },
            JammiError::Schema {
                table: "query".into(),
                column: "query".into(),
                expected: "4 dimensions".into(),
                actual: "5 dimensions".into(),
            },
        ] {
            let expected_text = engine.to_string();
            let arc = Arc::new(DataFusionError::External(Box::new(engine)));
            let _still_shared = Arc::clone(&arc);
            let got = plan_error("Search collect", DataFusionError::Shared(arc));
            assert_eq!(got.to_string(), expected_text, "{got:?}");
            assert!(!matches!(got, JammiError::Other(_)), "{got:?}");
        }
    }

    /// A typed engine error under `Context` / `Shared` is recovered as itself.
    #[test]
    fn engine_error_under_context_and_shared_is_recovered_typed() {
        let mk = || {
            DataFusionError::External(Box::new(JammiError::Unavailable {
                resource: "segment t/1".into(),
                reason: "unreachable".into(),
            }))
            .context("leaf")
        };
        assert!(matches!(
            plan_error("Search collect", mk()),
            JammiError::Unavailable { .. }
        ));
        let arc = Arc::new(mk());
        let _still_shared = Arc::clone(&arc);
        assert!(matches!(
            plan_error("Search collect", DataFusionError::Shared(arc)),
            JammiError::Unavailable { .. }
        ));
    }
}
