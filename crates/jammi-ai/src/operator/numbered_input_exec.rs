//! `NumberedInputExec` — the ordered, numbered, single-partition input every
//! model reads.
//!
//! It appends `_ordinal`: a non-null `UInt64`, contiguous from 0, in the order
//! the rows leave this node. Everything above it is a function of that column
//! — the forward chunks (`inference::chunk`), the exchange that fans a plan
//! out, and the merge that restores the sequence — so the order is fixed
//! here, once, before any of them.
//!
//! [`RowOrder`] names the order:
//!
//! * [`RowOrder::Keyed`] — the deterministic total order of a source scan.
//!   Null keys are counted over the whole input, the rows are sorted on
//!   `(CAST(key AS Utf8) ASC NULLS LAST, _content_hash ASC NULLS LAST)`, then
//!   numbered. Rows tied on both sort keys have equal `_row_id` and equal
//!   content, so they are mutually substitutable and the written bytes are
//!   invariant under permuting them — which a partial key could not promise,
//!   since the arrow sort is unstable and a coalesce interleaves
//!   nondeterministically. The cast is `safe: false`: a key that cannot
//!   render fails loudly, never nulls. A null key is one
//!   `JammiError::InvalidKey { column, null_count }` carrying the exact
//!   total, raised BEFORE any row is emitted: the sort is blocking, so it
//!   holds every row back until the count is complete, and the model above
//!   is never invoked.
//! * [`RowOrder::Arrival`] — the order rows arrive in, for an input with no
//!   key order to impose (an `annotate` over an arbitrary relation). Rows
//!   stream through as they are numbered.
//!
//! The node is OPAQUE: `children()` exposes only the input, and the
//! check → sort → number sequence is private to it. An optimizer rule can
//! move or parallelise only what it can see, so none can push the sort below
//! the null-key count (which would let rows through before the count is
//! complete) or split the count across partitions. The count and the
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
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::internal_err;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::expressions::{col, CastExpr};
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::Stream;
use jammi_db::store::schema::CONTENT_HASH_COLUMN;

use super::key_check_exec::KeyCheckExec;
use crate::inference::schema::ORDINAL_COLUMN;

/// The order [`NumberedInputExec`] numbers its rows in. See the module doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowOrder {
    /// The total order `(CAST(key AS Utf8), _content_hash)`, null keys
    /// refused. The input must carry `key_column` and `_content_hash`.
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

/// The ordered, numbered, single-partition input. See the module doc.
#[derive(Debug)]
pub struct NumberedInputExec {
    input: Arc<dyn ExecutionPlan>,
    order: RowOrder,
    /// `input` in `order`, unnumbered: what `execute` reads. Private, and
    /// absent from `children()`, so no rule can rewrite it.
    ordered: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl NumberedInputExec {
    /// Number `input` in `order`. Refuses an input that already carries
    /// `_ordinal`, and a keyed order whose key or `_content_hash` column the
    /// input lacks.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, order: RowOrder) -> DfResult<Self> {
        let input_schema = input.schema();
        if input_schema.field_with_name(ORDINAL_COLUMN).is_ok() {
            return Err(DataFusionError::Plan(format!(
                "NumberedInputExec: input schema already has an '{ORDINAL_COLUMN}' column"
            )));
        }
        let ordered = Self::ordered(Arc::clone(&input), &order)?;
        let schema: SchemaRef = Arc::new(Schema::new(
            input_schema
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .chain(std::iter::once(Field::new(
                    ORDINAL_COLUMN,
                    DataType::UInt64,
                    false,
                )))
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
            ordered,
            schema,
            properties: Arc::new(properties),
        })
    }

    /// The order this node numbers its rows in.
    pub fn order(&self) -> &RowOrder {
        &self.order
    }

    fn ordered(
        input: Arc<dyn ExecutionPlan>,
        order: &RowOrder,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let RowOrder::Keyed { key_column } = order else {
            return Ok(input);
        };
        let checked: Arc<dyn ExecutionPlan> = Arc::new(KeyCheckExec::try_new(input, key_column)?);
        let schema = checked.schema();
        let key: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
            col(key_column, schema.as_ref())?,
            DataType::Utf8,
            None,
        ));
        let hash = col(CONTENT_HASH_COLUMN, schema.as_ref())?;
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new(key, ASCENDING),
            PhysicalSortExpr::new(hash, ASCENDING),
        ])
        .ok_or_else(|| DataFusionError::Internal("the keyed ordering is empty".into()))?;
        Ok(Arc::new(SortExec::new(ordering, checked)))
    }
}

impl DisplayAs for NumberedInputExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        match &self.order {
            RowOrder::Keyed { key_column } => {
                write!(f, "NumberedInputExec: order=key({key_column})")
            }
            RowOrder::Arrival => write!(f, "NumberedInputExec: order=arrival"),
        }
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
        Ok(Box::pin(NumberedStream {
            inner: self.ordered.execute(0, context)?,
            schema: Arc::clone(&self.schema),
            next: 0,
        }))
    }
}

struct NumberedStream {
    inner: SendableRecordBatchStream,
    schema: SchemaRef,
    next: u64,
}

impl NumberedStream {
    fn number(&mut self, batch: &RecordBatch) -> DfResult<RecordBatch> {
        let end = self
            .next
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(|| DataFusionError::Internal("the row ordinal overflowed u64".into()))?;
        let ordinals: ArrayRef = Arc::new((self.next..end).collect::<UInt64Array>());
        self.next = end;
        let columns = batch
            .columns()
            .iter()
            .cloned()
            .chain(std::iter::once(ordinals))
            .collect();
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

impl Stream for NumberedStream {
    type Item = DfResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner)
            .poll_next(cx)
            .map(|item| item.map(|batch| batch.and_then(|b| self.number(&b))))
    }
}

impl RecordBatchStream for NumberedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use datafusion::config::ConfigOptions;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_optimizer::enforce_distribution::EnforceDistribution;
    use datafusion::physical_optimizer::enforce_sorting::EnforceSorting;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::common::collect;
    use datafusion::physical_plan::displayable;
    use datafusion::prelude::SessionContext;
    use futures::StreamExt;
    use jammi_db::error::JammiError;

    fn source_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
        ]))
    }

    fn batch(ids: Vec<Option<i64>>, hashes: Vec<&str>) -> RecordBatch {
        RecordBatch::try_new(
            source_schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
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

    fn keyed(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            NumberedInputExec::try_new(
                Arc::new(CoalescePartitionsExec::new(input)),
                RowOrder::Keyed {
                    key_column: "id".into(),
                },
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

    /// Keyed: every input partition's rows leave partition 0 in the total
    /// order `(CAST(key AS Utf8), _content_hash)` — `"10"` before `"2"` —
    /// numbered contiguously from 0.
    #[tokio::test]
    async fn keyed_rows_leave_in_key_order_numbered_from_zero() {
        let plan = keyed(source(vec![
            batch(vec![Some(2)], vec!["b"]),
            batch(vec![Some(10)], vec!["a"]),
            batch(vec![Some(1), Some(1)], vec!["z", "a"]),
            batch(vec![Some(3)], vec!["c"]),
        ]));
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let out = collect(plan.execute(0, SessionContext::new().task_ctx()).unwrap())
            .await
            .unwrap();
        let (mut ids, mut hashes, mut ordinals) = (Vec::new(), Vec::new(), Vec::new());
        for b in &out {
            ids.extend(column::<Int64Array>(b, "id").values().iter().copied());
            hashes.extend(
                column::<StringArray>(b, CONTENT_HASH_COLUMN)
                    .iter()
                    .map(|h| h.unwrap().to_string()),
            );
            ordinals.extend(
                column::<UInt64Array>(b, ORDINAL_COLUMN)
                    .values()
                    .iter()
                    .copied(),
            );
        }
        assert_eq!(ids, vec![1, 1, 10, 2, 3]);
        assert_eq!(hashes, vec!["a", "z", "a", "b", "c"]);
        assert_eq!(ordinals, vec![0, 1, 2, 3, 4]);
    }

    /// Arrival: rows keep the order they arrive in, and the ordinals run
    /// contiguously across batch boundaries.
    #[tokio::test]
    async fn arrival_rows_are_numbered_across_batches_in_arrival_order() {
        let batches = vec![
            batch(vec![Some(7), Some(3)], vec!["x", "y"]),
            batch(vec![Some(5)], vec!["z"]),
            batch(vec![Some(1), Some(9), Some(2)], vec!["p", "q", "r"]),
        ];
        let input = MemorySourceConfig::try_new_exec(&[batches], source_schema(), None).unwrap();
        let plan = NumberedInputExec::try_new(input, RowOrder::Arrival).unwrap();
        let out = collect(plan.execute(0, SessionContext::new().task_ctx()).unwrap())
            .await
            .unwrap();
        let ids: Vec<i64> = out
            .iter()
            .flat_map(|b| column::<Int64Array>(b, "id").values().to_vec())
            .collect();
        let ordinals: Vec<u64> = out
            .iter()
            .flat_map(|b| column::<UInt64Array>(b, ORDINAL_COLUMN).values().to_vec())
            .collect();
        assert_eq!(ids, vec![7, 3, 5, 1, 9, 2]);
        assert_eq!(ordinals, vec![0, 1, 2, 3, 4, 5]);
        let field = plan.schema().field_with_name(ORDINAL_COLUMN).cloned();
        assert_eq!(
            field.unwrap(),
            Field::new(ORDINAL_COLUMN, DataType::UInt64, false)
        );
    }

    /// An input that already carries `_ordinal` is refused by name, in either
    /// order, never silently overwritten or duplicated.
    #[test]
    fn an_input_already_carrying_the_ordinal_is_refused() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new(CONTENT_HASH_COLUMN, DataType::Utf8, true),
            Field::new(ORDINAL_COLUMN, DataType::UInt64, false),
        ]));
        for order in [
            RowOrder::Arrival,
            RowOrder::Keyed {
                key_column: "id".into(),
            },
        ] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None).unwrap();
            let err = NumberedInputExec::try_new(input, order)
                .expect_err("a colliding _ordinal must refuse");
            assert!(err.to_string().contains(ORDINAL_COLUMN), "{err}");
        }
    }

    /// The numbering is a total only over one stream, so the node refuses to
    /// execute over an input of several partitions rather than number one.
    #[test]
    fn an_input_of_several_partitions_refuses_to_execute() {
        let src = source(vec![
            batch(vec![Some(1)], vec!["a"]),
            batch(vec![Some(2)], vec!["b"]),
        ]);
        let plan = NumberedInputExec::try_new(src, RowOrder::Arrival).unwrap();
        let err = plan
            .execute(0, SessionContext::new().task_ctx())
            .err()
            .expect("a multi-partition input must refuse");
        assert!(err.to_string().contains("one partition"), "{err}");
    }

    /// A null key anywhere in the input is one typed refusal carrying the
    /// exact total, and it is the FIRST item the stream yields: no row is
    /// emitted before it.
    #[tokio::test]
    async fn keyed_null_keys_refuse_with_the_exact_total_before_any_row() {
        let plan = keyed(source(vec![
            batch(vec![Some(1), None], vec!["a", "b"]),
            batch(vec![None, None, Some(5)], vec!["c", "d", "e"]),
        ]));
        let mut stream = plan.execute(0, SessionContext::new().task_ctx()).unwrap();
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
            Ok(Box::pin(
                datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(
                    source_schema(),
                    futures::stream::iter(items.collect::<Vec<_>>()),
                ),
            ))
        }
    }

    /// An upstream error crosses the node as the typed variant it was raised
    /// as, in both orders.
    #[tokio::test]
    async fn an_upstream_error_passes_through_typed() {
        for order in [
            RowOrder::Arrival,
            RowOrder::Keyed {
                key_column: "id".into(),
            },
        ] {
            let src = Arc::new(FailingSource::new(vec![batch(vec![Some(1)], vec!["a"])]));
            let plan = NumberedInputExec::try_new(src, order.clone()).unwrap();
            let err = collect(plan.execute(0, SessionContext::new().task_ctx()).unwrap())
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
        let numbered = keyed(source(vec![
            batch(vec![Some(1), None], vec!["a", "b"]),
            batch(vec![None], vec!["c"]),
            batch(vec![Some(3)], vec!["d"]),
            batch(vec![Some(4)], vec!["e"]),
        ]));
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

        let mut stream = optimized
            .execute(0, SessionContext::new().task_ctx())
            .unwrap();
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
