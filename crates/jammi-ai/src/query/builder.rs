use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{
    new_null_array, Array, ArrayRef, AsArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{JoinType, NullEquality};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::index::{Admission, FiniteQuery, SearchMethod};
use jammi_db::session::QueryContext;
use jammi_db::store::manifest::ProducingDescriptor;
use jammi_db::ChannelId;

use crate::evidence::{merge_channels, ChannelContribution};
use crate::index::{LexicalHit, LexicalIndex};
use crate::operator::vector_search_exec::VectorSearchExec;
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
    /// `query`'s provenance decides whose fault a bad one is: a caller's
    /// vector fails as a schema-class error (`InvalidArgument` on the wire); a
    /// vector read back from storage (query-by-example) fails as a corrupt
    /// artifact named by its table. A non-finite component is refused before
    /// the caller hands the query here (by [`FiniteQuery::new`]); this is the
    /// EARLIEST point the width is checked: against the table's authority
    /// ([`jammi_db::store::ResultStore::query_width`]) before any plan node
    /// exists — so a query fault never reaches placement, the failure
    /// ladder, or a plan shipped to another process. Everything downstream
    /// holds a [`jammi_db::index::ValidatedQuery`] and checks only its own
    /// artifact's width.
    ///
    /// With a `filter`, the search ranks only the source rows the filter
    /// selects ([`admitted_rows`]): the `k` nearest of those, however few the
    /// filter selects — never the `k` nearest overall, filtered afterwards.
    pub(crate) async fn new(
        session: Arc<InferenceSession>,
        source_id: &str,
        query: FiniteQuery,
        k: usize,
        embedding_table: Option<&str>,
        method: SearchMethod,
        filter: Option<&str>,
    ) -> Result<Self> {
        let table = session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let result_store = session.result_store();
        let width = result_store.query_width(session.context(), &table).await?;
        let query_vec = query.against_authority(width)?;
        let admission = admitted_rows(&session, &table, filter).await?;

        let ann = VectorSearchExec::new(
            table.clone(),
            query_vec,
            k,
            method,
            admission,
            result_store,
            session.context().clone(),
        )?;
        Self::ranked(
            session,
            &table,
            Arc::new(ann),
            Ranking::SIMILARITY,
            "vector",
        )
        .await
    }

    /// Start a lexical (BM25) search over a source's lexical table.
    ///
    /// Resolves the named lexical table (or the source's newest) through the
    /// tenant-scoped catalog, ranks its rows against `text` with the table
    /// version's BM25 index — built from the version's rows the first time
    /// this process searches it — and hydrates the top `k` from the source
    /// like a dense search. Each row carries the `bm25` channel's
    /// `bm25_score` and 0-based `bm25_rank`. With a `filter`, only the source
    /// rows it selects are ranked, as for a dense search.
    pub(crate) async fn lexical(
        session: Arc<InferenceSession>,
        source_id: &str,
        text: &str,
        k: usize,
        lexical_table: Option<&str>,
        filter: Option<&str>,
    ) -> Result<Self> {
        let table = session
            .catalog()
            .resolve_lexical_table(source_id, lexical_table)
            .await?;
        let pin = session.result_store().pin_current_version(table).await?;
        let index = session
            .lexical_indexes()
            .get_or_build(pin.input_anchor(), build_lexical_index(&session, &pin))
            .await?;
        let admission = admitted_rows(&session, pin.record(), filter).await?;
        let hits = index.search(text, k, &admission)?;
        let plan = lexical_hits_plan(&pin.record().source_id, hits)?;
        Self::ranked(session, pin.record(), plan, Ranking::BM25_RANK, "bm25").await
    }

    /// Hydrate a ranked plan — `_row_id`, `_source_id` and its channel's score
    /// columns — by joining it back to `table`'s source on the table's key
    /// column, in `ranking` order, contributing on `channel`.
    ///
    /// The ranked rows' key is a string; the source key column may be another
    /// type, so it is cast to Utf8, as are the source's string columns, to
    /// avoid Utf8View/Utf8 mismatches from the Parquet reader.
    async fn ranked(
        session: Arc<InferenceSession>,
        table: &ResultTableRecord,
        mut plan: Arc<dyn ExecutionPlan>,
        ranking: Ranking,
        channel: &str,
    ) -> Result<Self> {
        if let Some(ref key_col) = table.key_column {
            let relation = session.input_relation(&table.source_id).await?;
            let source_cols =
                build_hydration_select(session.context(), &relation, key_col, &plan.schema())
                    .await?;
            let sql = format!("SELECT {source_cols} FROM {relation}");
            let df = session
                .context()
                .sql(&sql)
                .await
                .map_err(|e| JammiError::from(e.context("Hydration scan")))?;
            let source_plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
                df.create_physical_plan()
                    .await
                    .map_err(|e| JammiError::from(e.context("Hydration plan")))?,
            ));

            let left_col = col("_row_id", plan.schema().as_ref())
                .map_err(|e| JammiError::from(e.context("Hydration join")))?;
            let right_col = col("_join_key", source_plan.schema().as_ref())
                .map_err(|e| JammiError::from(e.context("Hydration join")))?;

            let join = HashJoinExec::try_new(
                plan,
                source_plan,
                vec![(left_col, right_col)],
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .map_err(|e| JammiError::from(e.context("Hydration join")))?;

            // Drop the _join_key column (redundant with _row_id)
            plan = drop_column(Arc::new(join), "_join_key")?;

            // Re-sort into the ranking (the join does not preserve order).
            let rank_col = col(ranking.column, plan.schema().as_ref())
                .map_err(|e| JammiError::from(e.context("Hydration sort")))?;
            let sort_expr = datafusion::physical_expr::PhysicalSortExpr {
                expr: rank_col,
                options: arrow::compute::SortOptions {
                    descending: ranking.descending,
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
            channels: vec![ChannelId::new(channel)?],
            annotated: false,
        })
    }

    /// Filter results with a SQL predicate.
    pub fn filter(mut self, predicate: &str) -> Result<Self> {
        let arrow_schema = self.plan.schema();
        let df_schema = datafusion::common::DFSchema::try_from(arrow_schema.as_ref().clone())
            .map_err(|e| JammiError::from(e.context("DFSchema conversion")))?;
        let expr = datafusion::prelude::SessionContext::new()
            .parse_sql_expr(predicate, &df_schema)
            .map_err(|e| invalid("filter", e))?;
        let execution_props = datafusion::execution::context::ExecutionProps::new();
        let physical_expr =
            datafusion::physical_expr::create_physical_expr(&expr, &df_schema, &execution_props)
                .map_err(|e| invalid("filter", e))?;
        let filter =
            datafusion::physical_plan::filter::FilterExec::try_new(physical_expr, self.plan)
                .map_err(|e| invalid("filter", e))?;
        self.plan = Arc::new(filter);
        Ok(self)
    }

    /// Sort results by a column.
    pub fn sort(mut self, column: &str, descending: bool) -> Result<Self> {
        let col_expr =
            datafusion::physical_expr::expressions::col(column, self.plan.schema().as_ref())
                .map_err(|e| invalid("sort", e))?;
        let sort_expr = datafusion::physical_expr::PhysicalSortExpr {
            expr: col_expr,
            options: arrow::compute::SortOptions {
                descending,
                nulls_first: false,
            },
        };
        let ordering =
            datafusion::physical_expr::LexOrdering::new(vec![sort_expr]).ok_or_else(|| {
                JammiError::Internal("a one-column sort has an empty ordering".into())
            })?;
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
                let expr = col(name, schema.as_ref()).map_err(|e| invalid("select", e))?;
                Ok((expr as Arc<dyn PhysicalExpr>, name.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = ProjectionExec::try_new(exprs, self.plan)
            .map_err(|e| JammiError::from(e.context("ProjectionExec")))?;
        self.plan = Arc::new(projection);
        Ok(self)
    }

    /// Join search results with another registered source.
    ///
    /// `on` is `"left_col=right_col"`. `how` is `"inner"` or `"left"` (default).
    pub async fn join(mut self, source: &str, on: &str, how: Option<&str>) -> Result<Self> {
        let (left_col_name, right_col_name) =
            on.split_once('=')
                .ok_or_else(|| JammiError::InvalidArgument {
                    argument: "on".into(),
                    reason: format!("a join condition is `left = right`, got '{on}'"),
                })?;

        let join_type = match how {
            Some("inner") => JoinType::Inner,
            _ => JoinType::Left,
        };

        let table = self.session.first_table(source).await?;
        let sql = format!("SELECT * FROM {table}");
        let df = self
            .session
            .context()
            .sql(&sql)
            .await
            .map_err(|e| JammiError::from(e.context("Join")))?;
        let right_plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
            df.create_physical_plan()
                .await
                .map_err(|e| JammiError::from(e.context("Join")))?,
        ));

        let left_col =
            col(left_col_name, self.plan.schema().as_ref()).map_err(|e| invalid("on", e))?;
        let right_col =
            col(right_col_name, right_plan.schema().as_ref()).map_err(|e| invalid("on", e))?;

        let join = HashJoinExec::try_new(
            self.plan,
            right_plan,
            vec![(left_col, right_col)],
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .map_err(|e| JammiError::from(e.context("Join")))?;

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
        task: jammi_datafusion::ModelTask,
        columns: &[String],
    ) -> Result<Self> {
        let model_source = jammi_datafusion::ModelSource::parse(model);
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
            .map_err(|e| JammiError::from(e.context("Search execute")))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| JammiError::from(e.context("Search collect")))?;

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
/// A channel contributes when any of its declared columns is present in the
/// batch under its declared name; a declared column the batch lacks (a
/// selection kept only some of the channel's columns) contributes all-null. A
/// channel with none present produces no contribution, and its declared
/// columns become all-null in the merged output. Every declared column is
/// stripped from the batch either way, so none appears twice. Dtype mismatches
/// are not coerced here; `merge_channels`'s validator surfaces them as a typed
/// `ChannelAssembly` error so callers see the real mismatch.
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
        let positions: Vec<Option<usize>> = spec
            .columns
            .iter()
            .map(|c| batch.schema().index_of(&c.name).ok())
            .collect();
        if positions.iter().all(Option::is_none) {
            continue;
        }
        let columns: Vec<ArrayRef> = positions
            .iter()
            .zip(&spec.columns)
            .map(|(position, declared)| match position {
                Some(i) => Arc::clone(batch.column(*i)),
                None => new_null_array(&declared.data_type.to_arrow(), batch.num_rows()),
            })
            .collect();
        to_remove.extend(positions.into_iter().flatten());
        contributions.push(ChannelContribution {
            channel: id.clone(),
            columns,
        });
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
/// The rows of `table`'s source a search may return: every row with no
/// `filter`; with one, the rows the filter selects, by the key the search
/// results carry as `_row_id` (the same `Utf8` cast the hydration join
/// matches on). Evaluated once, over the source, under the session's tenant
/// scope — the admission the index then searches within.
async fn admitted_rows(
    session: &InferenceSession,
    table: &ResultTableRecord,
    filter: Option<&str>,
) -> Result<Admission> {
    let Some(filter) = filter else {
        return Ok(Admission::Every);
    };
    let key_col = table
        .key_column
        .as_deref()
        .ok_or_else(|| JammiError::Schema {
            table: table.table_name.clone(),
            column: "<key_column>".to_string(),
            expected: "a recorded key column, which a filter's rows are matched to the search's by"
                .to_string(),
            actual: "none".to_string(),
        })?;
    let relation = session.input_relation(&table.source_id).await?;
    let batches = session
        .context()
        .sql(&format!(
            "SELECT arrow_cast(\"{key_col}\", 'Utf8') AS _admitted FROM {relation} WHERE {filter}"
        ))
        .await?
        .collect()
        .await?;
    let mut rows = Vec::new();
    for batch in &batches {
        let keys = batch.column(0).as_string::<i32>();
        rows.extend(keys.iter().flatten().map(str::to_string));
    }
    Ok(Admission::rows(rows))
}

async fn build_hydration_select(
    ctx: &QueryContext,
    table: &jammi_db::source::InputRelation,
    key_col: &str,
    result: &arrow::datatypes::Schema,
) -> Result<String> {
    let df = ctx
        .sql(&format!("SELECT * FROM {table} LIMIT 0"))
        .await
        .map_err(|e| JammiError::from(e.context("Schema introspection")))?;
    let schema = df.schema();

    let mut cols = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        // A source column named like one of the search result's own columns
        // cannot be emitted beside it. The key column is carried by the
        // result's `_row_id`; any other such column is refused, never
        // silently shadowed.
        if result.field_with_name(name).is_ok() {
            if name == key_col {
                continue;
            }
            return Err(JammiError::Schema {
                table: table.to_string(),
                column: name.to_string(),
                expected: "a source column named unlike the search result's own columns"
                    .to_string(),
                actual: format!("`{name}`, which the search result already carries"),
            });
        }
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
                .map_err(|e| JammiError::from(e.context("drop_column")))?;
            Ok((expr as Arc<dyn PhysicalExpr>, f.name().to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let projection = ProjectionExec::try_new(exprs, plan)
        .map_err(|e| JammiError::from(e.context("drop_column projection")))?;
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
/// The column a ranked plan is ordered by, and its direction.
struct Ranking {
    column: &'static str,
    descending: bool,
}

impl Ranking {
    /// A dense search: most similar first.
    const SIMILARITY: Self = Self {
        column: "similarity",
        descending: true,
    };
    /// A lexical search: rank 0 first — the index's own order, ties broken.
    const BM25_RANK: Self = Self {
        column: "bm25_rank",
        descending: false,
    };
}

/// Build the BM25 index of the lexical table version `pin` names from its
/// `(_row_id, text)` rows, under the analyzer its producing descriptor records.
async fn build_lexical_index(
    session: &InferenceSession,
    pin: &jammi_db::store::PinnedSource,
) -> Result<LexicalIndex> {
    let result_store = session.result_store();
    let analyzer = match result_store.producing_descriptor(pin).await? {
        ProducingDescriptor::LexicalIndex { analyzer, .. } => analyzer,
        other => {
            return Err(JammiError::Lexical(format!(
                "table '{}' records a {other:?} descriptor, not a lexical index",
                pin.table_name()
            )))
        }
    };
    let ctx = session.context();
    let batches = ctx
        .read_table(result_store.pinned_provider(ctx, pin).await?)
        .map_err(JammiError::from)?
        .select_columns(&["_row_id", "text"])
        .map_err(JammiError::from)?
        .collect()
        .await
        .map_err(JammiError::from)?;
    let mut rows: Vec<(String, String)> = Vec::new();
    for batch in &batches {
        let row_ids = utf8_column(batch, "_row_id", pin.table_name())?;
        let texts = utf8_column(batch, "text", pin.table_name())?;
        for i in 0..batch.num_rows() {
            if row_ids.is_null(i) {
                return Err(JammiError::Schema {
                    table: pin.table_name().to_string(),
                    column: "_row_id".into(),
                    expected: "a key on every row".into(),
                    actual: "null".into(),
                });
            }
            let text = if texts.is_null(i) { "" } else { texts.value(i) };
            rows.push((row_ids.value(i).to_string(), text.to_string()));
        }
    }
    LexicalIndex::build(rows, analyzer)
}

/// `column` of `batch` as a Utf8 array (the Parquet reader may hand back a
/// `Utf8View`).
fn utf8_column(batch: &RecordBatch, column: &str, table: &str) -> Result<StringArray> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| JammiError::Schema {
            table: table.to_string(),
            column: column.to_string(),
            expected: "Utf8".into(),
            actual: "missing".into(),
        })?;
    let cast = arrow::compute::cast(array, &DataType::Utf8).map_err(|e| JammiError::Schema {
        table: table.to_string(),
        column: column.to_string(),
        expected: "Utf8".into(),
        actual: format!("{:?} ({e})", array.data_type()),
    })?;
    Ok(cast.as_string::<i32>().clone())
}

/// The lexical hits as a ranked plan: `_row_id`, `_source_id`, and the `bm25`
/// channel's `bm25_score` and `bm25_rank`, in rank order.
fn lexical_hits_plan(source_id: &str, hits: Vec<LexicalHit>) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("_source_id", DataType::Utf8, false),
        Field::new("bm25_score", DataType::Float32, false),
        Field::new("bm25_rank", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values(
                hits.iter().map(|h| h.row_id.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                hits.iter().map(|_| source_id),
            )),
            Arc::new(Float32Array::from_iter_values(
                hits.iter().map(|h| h.bm25_score),
            )),
            Arc::new(Int64Array::from_iter_values(
                hits.iter().map(|h| h.rank as i64),
            )),
        ],
    )
    .map_err(|e| JammiError::Lexical(format!("lexical hits batch: {e}")))?;
    Ok(MemorySourceConfig::try_new_exec(
        &[vec![batch]],
        schema,
        None,
    )?)
}

/// The caller's `argument` — a predicate, a column, a join condition — did
/// not plan against the rows it names.
fn invalid(argument: &str, e: datafusion::error::DataFusionError) -> JammiError {
    JammiError::InvalidArgument {
        argument: argument.into(),
        reason: e.to_string(),
    }
}
