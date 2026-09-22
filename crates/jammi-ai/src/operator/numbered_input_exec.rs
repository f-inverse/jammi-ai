//! `NumberedInputExec` — the ordered, numbered, chunked, single-partition
//! input every model reads.
//!
//! It appends two non-null `UInt64` columns in the order the rows leave this
//! node: `_ordinal`, contiguous from 0, and `_chunk`, the forward chunk each
//! row belongs to (`inference::chunk`). Everything above it is a function of
//! those columns — the forward chunks, the exchange that fans a plan out, and
//! the merge that restores the sequence — so the order and the chunks are
//! fixed here, once, before any of them.
//!
//! Every row's COST — its length along the axis the forward pads, the
//! model's own tokenisation for text, one for a fixed-shape input — is
//! appended as the rows stream in (`super::row_cost_exec`). [`RowOrder`]
//! names the order the rows then take:
//!
//! * [`RowOrder::Keyed`] — the deterministic total order of a source scan:
//!   `(_cost ASC, key ASC NULLS LAST, _content_hash ASC NULLS LAST)`, the
//!   key on its own type. Cost first is what makes the chunks cheap: rows
//!   that share a forward are then nearly equal in length, so the forward
//!   pads to little more than their real length instead of to the longest
//!   row of an arbitrary run (`jammi_numerics::batch_shape`). Rows tied on
//!   all three sort keys have equal `_row_id` and equal content, so they are
//!   mutually substitutable and the written bytes are invariant under
//!   permuting them — which a partial key could not promise, since the
//!   arrow sort is unstable and a coalesce interleaves nondeterministically.
//!   A key of a type that cannot render as `_row_id` (a struct) is refused
//!   at plan build. Null keys are counted over the whole input: a null key
//!   is one `JammiError::InvalidKey { column, null_count }` carrying the
//!   exact total, raised BEFORE any row is emitted — the sort is blocking,
//!   so it holds every row back until the count is complete, and no forward
//!   ever runs over the input.
//! * [`RowOrder::Arrival`] — the order rows arrive in, for an input with no
//!   key order to impose (an `annotate` over an arbitrary relation). Rows
//!   stream through as they are numbered and chunked.
//!
//! In either order the chunks are one pass of a
//! [`ChunkCutter`] over the cost sequence under the spec's
//! [`ChunkBudget`](jammi_numerics::ChunkBudget), so the chunk of
//! a row is a function of the ordered rows alone: identical at every
//! partition count, under every re-batching, and on every executor.
//!
//! The node is OPAQUE: `children()` exposes only the input, and the
//! cost → check → sort → number sequence is private to it. An optimizer rule
//! can move or parallelise only what it can see, so none can push the sort
//! below the null-key count (which would let rows through before the count
//! is complete) or split the count across partitions. The count and the
//! numbering are totals only if one stream sees every row, so the node
//! REQUIRES a single input partition ([`Distribution::SinglePartition`]) and
//! refuses to execute over more.
//!
//! An upstream error passes through as the OWNED `DataFusionError` it arrived
//! as — never re-wrapped, never stringified — so a typed refusal raised below
//! this node reaches the caller as that same variant.
//!
//! It never overrides `supports_limit_pushdown` (default `false`) nor
//! `with_fetch` (default `None`): a fetch stays above a node that must see
//! every row to count and order them.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::internal_err;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::{EquivalenceProperties, LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::{StreamExt, TryStreamExt};
use jammi_db::store::schema::CONTENT_HASH_COLUMN;
use jammi_numerics::ChunkCutter;

use super::inference_exec::{InferenceRuntime, InferenceSpec};
use super::key_check_exec::KeyCheckExec;
use super::row_cost_exec::{RowCostExec, COST_COLUMN};
use crate::inference::chunk::CHUNK_COLUMN;
use crate::inference::schema::ORDINAL_COLUMN;

/// The order [`NumberedInputExec`] numbers its rows in. See the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowOrder {
    /// The total order `(_cost, key, _content_hash)`, null keys refused. The
    /// input must carry `key_column` and `_content_hash`.
    Keyed {
        /// The row-identity column.
        key_column: String,
    },
    /// The order rows arrive in.
    Arrival,
}

/// `ASC NULLS LAST` — the direction of every ordering this engine declares
/// over `_ordinal` and the keyed sort.
const ASCENDING: SortOptions = SortOptions {
    descending: false,
    nulls_first: false,
};

/// The ordering `[_ordinal ASC]` over `schema`, which must carry the column.
pub fn ordinal_ordering(schema: &Schema) -> DfResult<LexOrdering> {
    let ordinal = col(ORDINAL_COLUMN, schema)?;
    LexOrdering::new([PhysicalSortExpr::new(ordinal, ASCENDING)])
        .ok_or_else(|| DataFusionError::Internal("an ordering of one expression is empty".into()))
}

/// The ordered, numbered, chunked, single-partition input. See the module
/// doc.
pub struct NumberedInputExec {
    input: Arc<dyn ExecutionPlan>,
    order: RowOrder,
    spec: InferenceSpec,
    runtime: InferenceRuntime,
    /// `input` costed and in `order`, unnumbered: what `execute` reads.
    /// Private, and absent from `children()`, so no rule can rewrite it.
    ordered: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for NumberedInputExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumberedInputExec")
            .field("order", &self.order)
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl NumberedInputExec {
    /// Number `input` in `order`, chunked under `spec.chunk` by the costs of
    /// `spec`'s model and task. Refuses an input that already carries
    /// `_ordinal` or `_chunk`, and a keyed order whose key or
    /// `_content_hash` column the input lacks or whose key cannot render as
    /// `_row_id`.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        order: RowOrder,
        spec: InferenceSpec,
        runtime: InferenceRuntime,
    ) -> DfResult<Self> {
        let input_schema = input.schema();
        for name in [ORDINAL_COLUMN, CHUNK_COLUMN] {
            if input_schema.field_with_name(name).is_ok() {
                return Err(DataFusionError::Plan(format!(
                    "NumberedInputExec: input schema already has a '{name}' column"
                )));
            }
        }
        let ordered = Self::ordered(Arc::clone(&input), &order, &spec, &runtime)?;
        let numbered = [ORDINAL_COLUMN, CHUNK_COLUMN]
            .into_iter()
            .map(|name| Field::new(name, DataType::UInt64, false));
        let schema: SchemaRef = Arc::new(Schema::new(
            input_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .chain(numbered)
                .collect::<Vec<_>>(),
        ));
        let mut eq = EquivalenceProperties::new(Arc::clone(&schema));
        eq.add_ordering(ordinal_ordering(schema.as_ref())?);
        let properties = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(1),
            ordered.pipeline_behavior(),
            ordered.boundedness(),
        );
        Ok(Self {
            input,
            order,
            spec,
            runtime,
            ordered,
            schema,
            properties: Arc::new(properties),
        })
    }

    /// The order this node numbers its rows in.
    pub fn order(&self) -> &RowOrder {
        &self.order
    }

    /// The model, task and chunk budget the rows are costed and chunked by.
    pub fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn ordered(
        input: Arc<dyn ExecutionPlan>,
        order: &RowOrder,
        spec: &InferenceSpec,
        runtime: &InferenceRuntime,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let RowOrder::Keyed { key_column } = order else {
            return Ok(Arc::new(RowCostExec::try_new(
                input,
                spec.clone(),
                runtime.clone(),
            )?));
        };
        let key_type = input
            .schema()
            .field_with_name(key_column)?
            .data_type()
            .clone();
        if !arrow::compute::can_cast_types(&key_type, &DataType::Utf8) {
            return Err(DataFusionError::Plan(format!(
                "NumberedInputExec: key column '{key_column}' (type {key_type:?}) cannot be \
                 rendered as _row_id"
            )));
        }
        let checked: Arc<dyn ExecutionPlan> = Arc::new(KeyCheckExec::try_new(input, key_column)?);
        let costed: Arc<dyn ExecutionPlan> = Arc::new(RowCostExec::try_new(
            checked,
            spec.clone(),
            runtime.clone(),
        )?);
        let schema = costed.schema();
        let ordering = LexOrdering::new(
            [COST_COLUMN, key_column.as_str(), CONTENT_HASH_COLUMN]
                .into_iter()
                .map(|name| col(name, schema.as_ref()).map(|c| PhysicalSortExpr::new(c, ASCENDING)))
                .collect::<DfResult<Vec<_>>>()?,
        )
        .ok_or_else(|| DataFusionError::Internal("the keyed ordering is empty".into()))?;
        Ok(Arc::new(SortExec::new(ordering, costed)))
    }
}

impl DisplayAs for NumberedInputExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match &self.order {
            RowOrder::Keyed { key_column } => {
                write!(f, "NumberedInputExec: order=key({key_column})")?;
            }
            RowOrder::Arrival => write!(f, "NumberedInputExec: order=arrival")?,
        }
        write!(
            f,
            ", batch_size={}, batch_tokens={}",
            self.spec.chunk.rows, self.spec.chunk.tokens
        )
    }
}

impl ExecutionPlan for NumberedInputExec {
    fn name(&self) -> &str {
        "NumberedInputExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![self.order == RowOrder::Arrival]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            self.order.clone(),
            self.spec.clone(),
            self.runtime.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input_partitions = self.input.output_partitioning().partition_count();
        if partition != 0 || input_partitions != 1 {
            return internal_err!(
                "NumberedInputExec numbers one partition: asked for partition {partition} of \
                 an input with {input_partitions}"
            );
        }
        let inner = self.ordered.execute(0, context)?;
        let cost_index = self.ordered.schema().index_of(COST_COLUMN)?;
        let schema = Arc::clone(&self.schema);
        let spec = self.spec.clone();
        let cache = Arc::clone(&self.runtime.model_cache);
        // The ladder the chunks are budgeted on is the model's own, so the
        // cutter is built once the model is loaded — a cache hit, the cost
        // node below having loaded it already.
        let numbered = async move {
            let ladder = cache
                .get_or_load(&spec.source, spec.task, spec.backend)
                .await
                .and_then(|guard| guard.model.shape_ladder(spec.task))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let mut numbering = Numbering {
                schema,
                cost_index,
                next: 0,
                cutter: ChunkCutter::new(spec.chunk, ladder),
            };
            Ok::<_, DataFusionError>(
                inner.map(move |batch| batch.and_then(|b| numbering.number(&b))),
            )
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::once(numbered).try_flatten(),
        )))
    }
}

/// The one sequential pass that numbers rows and cuts their chunks.
struct Numbering {
    schema: SchemaRef,
    cost_index: usize,
    next: u64,
    cutter: ChunkCutter,
}

impl Numbering {
    fn number(&mut self, batch: &RecordBatch) -> DfResult<RecordBatch> {
        let end = self
            .next
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| DataFusionError::Internal("the row ordinal overflowed u64".into()))?;
        let ordinals: ArrayRef = Arc::new((self.next..end).collect::<UInt64Array>());
        self.next = end;
        let costs = batch.column(self.cost_index);
        let costs = costs
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "the row cost column is {:?}, expected UInt32",
                    costs.data_type()
                ))
            })?;
        let chunks: ArrayRef = Arc::new(
            costs
                .values()
                .iter()
                .map(|&cost| self.cutter.push(cost))
                .collect::<UInt64Array>(),
        );
        let columns = batch
            .columns()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.cost_index)
            .map(|(_, c)| Arc::clone(c))
            .chain([ordinals, chunks])
            .collect();
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    use arrow::array::{Int64Array, StringArray};
    use datafusion::config::ConfigOptions;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_optimizer::enforce_distribution::EnforceDistribution;
    use datafusion::physical_optimizer::enforce_sorting::EnforceSorting;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::common::collect;
    use datafusion::physical_plan::displayable;
    use futures::StreamExt;
    use jammi_db::error::JammiError;
    use jammi_db::store::manifest::ComputeDeviceKind;
    use jammi_numerics::ChunkBudget;

    use crate::model::{ModelSource, ModelTask};
    use crate::session::InferenceSession;

    fn source_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("text", DataType::Utf8, true),
            Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
        ]))
    }

    /// Rows keyed `ids`, hashed `hashes`, each with `words` in-vocabulary
    /// words of text — its cost, plus the two special tokens.
    fn batch(ids: Vec<Option<i64>>, hashes: Vec<&str>, words: Vec<usize>) -> RecordBatch {
        RecordBatch::try_new(
            source_schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from_iter_values(
                    words.iter().map(|w| vec!["a"; *w].join(" ")),
                )),
                Arc::new(StringArray::from(hashes)),
            ],
        )
        .unwrap()
    }

    /// One partition per batch.
    fn source(parts: Vec<RecordBatch>) -> Arc<dyn ExecutionPlan> {
        let partitions: Vec<Vec<RecordBatch>> = parts.into_iter().map(|b| vec![b]).collect();
        MemorySourceConfig::try_new_exec(&partitions, source_schema(), None).unwrap()
    }

    async fn session() -> (InferenceSession, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let session = InferenceSession::new(jammi_test_utils::test_config(dir.path()))
            .await
            .unwrap();
        (session, dir)
    }

    fn spec(rows: usize, tokens: usize) -> InferenceSpec {
        InferenceSpec {
            source: ModelSource::Local(jammi_test_utils::cookbook_fixture("tiny_bert")),
            task: ModelTask::TextEmbedding,
            content_columns: vec!["text".into()],
            key_column: "id".into(),
            source_id: "numbered-input".into(),
            backend: None,
            chunk: ChunkBudget {
                rows: NonZeroUsize::new(rows).unwrap(),
                tokens: NonZeroUsize::new(tokens).unwrap(),
            },
            embedding_dim: Some(32),
            regression_form: None,
            passthrough: Vec::new(),
            device_kind: ComputeDeviceKind::Cpu,
            partitions: NonZeroUsize::MIN,
        }
    }

    fn keyed(
        session: &InferenceSession,
        input: Arc<dyn ExecutionPlan>,
        rows: usize,
        tokens: usize,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            NumberedInputExec::try_new(
                Arc::new(CoalescePartitionsExec::new(input)),
                RowOrder::Keyed {
                    key_column: "id".into(),
                },
                spec(rows, tokens),
                session.inference_runtime(),
            )
            .unwrap(),
        )
    }

    fn column<'a, A: 'static>(batch: &'a RecordBatch, name: &str) -> &'a A {
        batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<A>()
            .unwrap()
    }

    fn u64s(out: &[RecordBatch], name: &str) -> Vec<u64> {
        out.iter()
            .flat_map(|b| column::<UInt64Array>(b, name).values().to_vec())
            .collect()
    }

    /// Keyed: every input partition's rows leave partition 0 in the total
    /// order `(cost, key, _content_hash)` — the two-word rows first, then
    /// by key on its own type (`2` before `10`) — numbered contiguously from
    /// 0 and chunked under the budget, with the private cost column gone.
    #[tokio::test]
    async fn keyed_rows_leave_in_cost_then_key_order_numbered_and_chunked() {
        let (session, _dir) = session().await;
        let plan = keyed(
            &session,
            source(vec![
                batch(vec![Some(2)], vec!["b"], vec![9]),
                batch(vec![Some(10)], vec!["a"], vec![2]),
                batch(vec![Some(1), Some(1)], vec!["z", "a"], vec![9, 9]),
                batch(vec![Some(3)], vec!["c"], vec![2]),
            ]),
            2,
            4096,
        );
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        assert!(plan.schema().field_with_name(COST_COLUMN).is_err());
        let out = collect(plan.execute(0, session.context().task_ctx()).unwrap())
            .await
            .unwrap();
        let ids: Vec<i64> = out
            .iter()
            .flat_map(|b| column::<Int64Array>(b, "id").values().to_vec())
            .collect();
        let hashes: Vec<String> = out
            .iter()
            .flat_map(|b| {
                column::<StringArray>(b, CONTENT_HASH_COLUMN)
                    .iter()
                    .map(|h| h.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(ids, vec![3, 10, 1, 1, 2]);
        assert_eq!(hashes, vec!["c", "a", "a", "z", "b"]);
        assert_eq!(u64s(&out, ORDINAL_COLUMN), vec![0, 1, 2, 3, 4]);
        assert_eq!(u64s(&out, CHUNK_COLUMN), vec![0, 0, 1, 1, 2]);
    }

    /// The token budget cuts the chunks: nine-word rows cost 11 tokens and
    /// pad to 16, so a budget of 40 tokens holds two of them, where a budget
    /// of 4096 under a row cap of 3 holds three.
    #[tokio::test]
    async fn the_chunk_budget_cuts_by_padded_tokens_and_by_rows() {
        let (session, _dir) = session().await;
        let rows = || {
            source(vec![batch(
                (1..=6).map(Some).collect(),
                vec!["a", "b", "c", "d", "e", "f"],
                vec![9; 6],
            )])
        };
        let by_tokens = keyed(&session, rows(), 100, 40);
        let out = collect(by_tokens.execute(0, session.context().task_ctx()).unwrap())
            .await
            .unwrap();
        assert_eq!(u64s(&out, CHUNK_COLUMN), vec![0, 0, 1, 1, 2, 2]);
        let by_rows = keyed(&session, rows(), 3, 4096);
        let out = collect(by_rows.execute(0, session.context().task_ctx()).unwrap())
            .await
            .unwrap();
        assert_eq!(u64s(&out, CHUNK_COLUMN), vec![0, 0, 0, 1, 1, 1]);
    }

    /// Arrival: rows keep the order they arrive in, the ordinals run
    /// contiguously across batch boundaries, and the chunks are cut in that
    /// order.
    #[tokio::test]
    async fn arrival_rows_are_numbered_across_batches_in_arrival_order() {
        let (session, _dir) = session().await;
        let batches = vec![
            batch(vec![Some(7), Some(3)], vec!["x", "y"], vec![1, 1]),
            batch(vec![Some(5)], vec!["z"], vec![1]),
            batch(
                vec![Some(1), Some(9), Some(2)],
                vec!["p", "q", "r"],
                vec![1, 1, 1],
            ),
        ];
        let input = MemorySourceConfig::try_new_exec(&[batches], source_schema(), None).unwrap();
        let plan = NumberedInputExec::try_new(
            input,
            RowOrder::Arrival,
            spec(4, 4096),
            session.inference_runtime(),
        )
        .unwrap();
        let out = collect(plan.execute(0, session.context().task_ctx()).unwrap())
            .await
            .unwrap();
        let ids: Vec<i64> = out
            .iter()
            .flat_map(|b| column::<Int64Array>(b, "id").values().to_vec())
            .collect();
        assert_eq!(ids, vec![7, 3, 5, 1, 9, 2]);
        assert_eq!(u64s(&out, ORDINAL_COLUMN), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(u64s(&out, CHUNK_COLUMN), vec![0, 0, 0, 0, 1, 1]);
        for name in [ORDINAL_COLUMN, CHUNK_COLUMN] {
            let field = plan.schema().field_with_name(name).cloned();
            assert_eq!(field.unwrap(), Field::new(name, DataType::UInt64, false));
        }
    }

    /// An input that already carries `_ordinal` or `_chunk` is refused by
    /// name, in either order, never silently overwritten or duplicated.
    #[tokio::test]
    async fn an_input_already_numbered_is_refused() {
        let (session, _dir) = session().await;
        for name in [ORDINAL_COLUMN, CHUNK_COLUMN] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("text", DataType::Utf8, true),
                Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
                Field::new(name, DataType::UInt64, false),
            ]));
            for order in [
                RowOrder::Arrival,
                RowOrder::Keyed {
                    key_column: "id".into(),
                },
            ] {
                let input =
                    MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None).unwrap();
                let err = NumberedInputExec::try_new(
                    input,
                    order,
                    spec(4, 4096),
                    session.inference_runtime(),
                )
                .expect_err("a colliding numbered column must refuse");
                assert!(err.to_string().contains(name), "{err}");
            }
        }
    }

    /// A key whose type has no `_row_id` rendering (a struct) is refused at
    /// plan build, naming the column and its type.
    #[tokio::test]
    async fn a_key_that_cannot_render_as_row_id_is_refused_at_plan_build() {
        let (session, _dir) = session().await;
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "id",
                DataType::Struct(vec![Field::new("a", DataType::Int32, false)].into()),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
            Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
        ]));
        let input = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        let err = NumberedInputExec::try_new(
            input,
            RowOrder::Keyed {
                key_column: "id".into(),
            },
            spec(4, 4096),
            session.inference_runtime(),
        )
        .expect_err("a struct key must refuse");
        assert!(err.to_string().contains("'id'"), "{err}");
        assert!(err.to_string().contains("Struct"), "{err}");
    }

    /// The numbering is a total only over one stream, so the node refuses to
    /// execute over an input of several partitions rather than number one.
    #[tokio::test]
    async fn an_input_of_several_partitions_refuses_to_execute() {
        let (session, _dir) = session().await;
        let src = source(vec![
            batch(vec![Some(1)], vec!["a"], vec![1]),
            batch(vec![Some(2)], vec!["b"], vec![1]),
        ]);
        let plan = NumberedInputExec::try_new(
            src,
            RowOrder::Arrival,
            spec(4, 4096),
            session.inference_runtime(),
        )
        .unwrap();
        let err = plan
            .execute(0, session.context().task_ctx())
            .err()
            .expect("a multi-partition input must refuse");
        assert!(err.to_string().contains("one partition"), "{err}");
    }

    /// A null key anywhere in the input is one typed refusal carrying the
    /// exact total, and it is the FIRST item the stream yields: no row is
    /// emitted before it.
    #[tokio::test]
    async fn keyed_null_keys_refuse_with_the_exact_total_before_any_row() {
        let (session, _dir) = session().await;
        let plan = keyed(
            &session,
            source(vec![
                batch(vec![Some(1), None], vec!["a", "b"], vec![1, 1]),
                batch(
                    vec![None, None, Some(5)],
                    vec!["c", "d", "e"],
                    vec![1, 1, 1],
                ),
            ]),
            4,
            4096,
        );
        let mut stream = plan.execute(0, session.context().task_ctx()).unwrap();
        let first = stream
            .next()
            .await
            .expect("the stream yields the refusal")
            .expect_err("the first item is the refusal, never a batch");
        match JammiError::from(first) {
            JammiError::InvalidKey { column, null_count } => {
                assert_eq!(column, "id");
                assert_eq!(null_count, 3);
            }
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    /// A leaf that yields `ok` batches and then one error.
    #[derive(Debug)]
    struct FailingSource {
        ok: Vec<RecordBatch>,
        properties: Arc<PlanProperties>,
    }

    impl FailingSource {
        fn new(ok: Vec<RecordBatch>) -> Self {
            let properties = PlanProperties::new(
                EquivalenceProperties::new(source_schema()),
                Partitioning::UnknownPartitioning(1),
                datafusion::physical_plan::execution_plan::EmissionType::Incremental,
                datafusion::physical_plan::execution_plan::Boundedness::Bounded,
            );
            Self {
                ok,
                properties: Arc::new(properties),
            }
        }
    }

    impl DisplayAs for FailingSource {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
            write!(f, "FailingSource")
        }
    }

    impl ExecutionPlan for FailingSource {
        fn name(&self) -> &str {
            "FailingSource"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            &self.properties
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DfResult<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> DfResult<SendableRecordBatchStream> {
            let items = self.ok.iter().cloned().map(Ok).chain(std::iter::once(Err(
                DataFusionError::External(Box::new(JammiError::LeaseLost {
                    table: "numbered_input_source".into(),
                })),
            )));
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                source_schema(),
                futures::stream::iter(items.collect::<Vec<_>>()),
            )))
        }
    }

    /// An upstream error crosses the node as the typed variant it was raised
    /// as, in both orders.
    #[tokio::test]
    async fn an_upstream_error_passes_through_typed() {
        let (session, _dir) = session().await;
        for order in [
            RowOrder::Arrival,
            RowOrder::Keyed {
                key_column: "id".into(),
            },
        ] {
            let src = Arc::new(FailingSource::new(vec![batch(
                vec![Some(1)],
                vec!["a"],
                vec![1],
            )]));
            let plan = NumberedInputExec::try_new(
                src,
                order.clone(),
                spec(4, 4096),
                session.inference_runtime(),
            )
            .unwrap();
            let err = collect(plan.execute(0, session.context().task_ctx()).unwrap())
                .await
                .expect_err("the upstream error must surface");
            assert!(
                matches!(
                    JammiError::from(err),
                    JammiError::LeaseLost { ref table } if table == "numbered_input_source"
                ),
                "{order:?}: the error must keep its variant"
            );
        }
    }

    fn find<'a>(
        plan: &'a Arc<dyn ExecutionPlan>,
        name: &str,
    ) -> Option<&'a Arc<dyn ExecutionPlan>> {
        let mut stack = vec![plan];
        while let Some(node) = stack.pop() {
            if node.name() == name {
                return Some(node);
            }
            stack.extend(node.children());
        }
        None
    }

    /// `EnforceDistribution` and `EnforceSorting` parallelise a sort over a
    /// coalesce by pushing the sort below it, carrying every node that
    /// declares no distribution requirement down with it. Under a parent sort
    /// that invites exactly that, both rules leave this node over its
    /// single-partition input — the coalesce stays directly below it, the
    /// four source partitions below that — and the null-key refusal still
    /// precedes any output.
    #[tokio::test]
    async fn the_distribution_and_sorting_rules_leave_the_node_intact() {
        let (session, _dir) = session().await;
        let numbered = keyed(
            &session,
            source(vec![
                batch(vec![Some(1), None], vec!["a", "b"], vec![1, 1]),
                batch(vec![None], vec!["c"], vec![1]),
                batch(vec![Some(3)], vec!["d"], vec![1]),
                batch(vec![Some(4)], vec!["e"], vec![1]),
            ]),
            4,
            4096,
        );
        let by_id = LexOrdering::new([PhysicalSortExpr::new(
            col("id", numbered.schema().as_ref()).unwrap(),
            ASCENDING,
        )])
        .unwrap();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(by_id, numbered));

        let mut options = ConfigOptions::default();
        options.execution.target_partitions = 8;
        options.optimizer.repartition_sorts = true;
        let distributed = EnforceDistribution::new().optimize(plan, &options).unwrap();
        let optimized = EnforceSorting::new()
            .optimize(distributed, &options)
            .unwrap();
        let shape = displayable(optimized.as_ref()).indent(true).to_string();

        let node = find(&optimized, "NumberedInputExec")
            .unwrap_or_else(|| panic!("the node survives the rules:\n{shape}"));
        let below = node.children()[0];
        assert_eq!(below.name(), "CoalescePartitionsExec", "{shape}");
        assert_eq!(below.output_partitioning().partition_count(), 1, "{shape}");
        assert_eq!(
            below.children()[0].output_partitioning().partition_count(),
            4,
            "{shape}"
        );
        assert!(find(node, "SortExec").is_none(), "{shape}");
        assert!(find(node, "KeyCheckExec").is_none(), "{shape}");
        assert!(find(node, "RowCostExec").is_none(), "{shape}");

        let mut stream = optimized.execute(0, session.context().task_ctx()).unwrap();
        let first = stream
            .next()
            .await
            .expect("the stream yields the refusal")
            .expect_err("the refusal precedes any output");
        assert!(matches!(
            JammiError::from(first),
            JammiError::InvalidKey { ref column, null_count: 2 } if column == "id"
        ));
    }
}
