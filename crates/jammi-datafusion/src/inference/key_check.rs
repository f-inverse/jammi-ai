//! `KeyCheckExec` — the null-key refusal at the input edge.
//!
//! A single-partition passthrough: every batch flows through unchanged while
//! the node counts the nulls in the RAW key column, and at end of input it
//! yields exactly one `Err(External(Error::InvalidKey {
//! column, null_count }))` if the total is non-zero. The source is scanned
//! once and the count is exact.
//!
//! Two consumers. A scan that only classifies its rows runs it directly
//! ([`key_checked`]). A model-facing input composes it privately below a
//! blocking sort ([`crate::inference::numbered`]), which holds every row back
//! until the count is complete, so the refusal precedes any row and the model
//! is never invoked.
//!
//! The count is a total only if one stream sees every row, so the node
//! REQUIRES a single input partition ([`Distribution::SinglePartition`]). The
//! declaration is what keeps an optimizer pass honest: a rule that treats an
//! undeclared node as partition-transparent would otherwise push the node
//! below a coalesce and turn one exact count into a count per partition.
//!
//! It never overrides `supports_limit_pushdown` (default `false`) nor
//! `with_fetch` (default `None`): a fetch must stay above it, never be pushed
//! into a node that must see every row to count.

use std::fmt::{self, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::Error;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion::common::internal_err;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::Stream;

/// The null-key counting passthrough. See the module doc.
#[derive(Debug)]
pub struct KeyCheckExec {
    input: Arc<dyn ExecutionPlan>,
    key_column: String,
    key_index: usize,
    properties: Arc<PlanProperties>,
}

impl KeyCheckExec {
    /// Wrap `input`, checking `key_column` (which must exist in the input
    /// schema — a typed plan error otherwise).
    pub fn try_new(input: Arc<dyn ExecutionPlan>, key_column: &str) -> DfResult<Self> {
        let key_index = input.schema().index_of(key_column)?;
        let properties = PlanProperties::new(
            input.equivalence_properties().clone(),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            key_column: key_column.to_string(),
            key_index,
            properties: Arc::new(properties),
        })
    }

    /// The key column this node counts nulls in.
    pub fn key_column(&self) -> &str {
        &self.key_column
    }
}

/// `KeyCheckExec` over `plan` brought to one partition — the shape a
/// classifying scan (no model above it) uses.
pub fn key_checked(
    plan: Arc<dyn ExecutionPlan>,
    key_column: &str,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(KeyCheckExec::try_new(
        crate::inference::exec::single_partition(plan),
        key_column,
    )?))
}

impl DisplayAs for KeyCheckExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "KeyCheckExec: key={}", self.key_column)
    }
}

impl ExecutionPlan for KeyCheckExec {
    fn name(&self) -> &str {
        "KeyCheckExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            &self.key_column,
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
                "KeyCheckExec counts over one partition: asked for partition {partition} \
                 of an input with {input_partitions}"
            );
        }
        let inner = self.input.execute(0, context)?;
        Ok(Box::pin(KeyCheckStream {
            inner,
            schema: self.schema(),
            key_column: self.key_column.clone(),
            key_index: self.key_index,
            nulls: AtomicU64::new(0),
            done: false,
        }))
    }
}

struct KeyCheckStream {
    inner: SendableRecordBatchStream,
    schema: SchemaRef,
    key_column: String,
    key_index: usize,
    nulls: AtomicU64,
    done: bool,
}

impl Stream for KeyCheckStream {
    type Item = DfResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let n = batch.column(self.key_index).null_count() as u64;
                self.nulls.fetch_add(n, Ordering::Relaxed);
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.done = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.done = true;
                let total = self.nulls.load(Ordering::Relaxed);
                if total > 0 {
                    Poll::Ready(Some(Err(DataFusionError::External(Box::new(
                        Error::InvalidKey {
                            column: self.key_column.clone(),
                            null_count: total,
                        },
                    )))))
                } else {
                    Poll::Ready(None)
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for KeyCheckStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::common::collect;
    use datafusion::prelude::SessionContext;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn batch(ids: Vec<Option<i64>>, texts: Vec<&str>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap()
    }

    /// One partition per batch.
    fn source(parts: Vec<RecordBatch>) -> Arc<dyn ExecutionPlan> {
        let partitions: Vec<Vec<RecordBatch>> = parts.into_iter().map(|b| vec![b]).collect();
        MemorySourceConfig::try_new_exec(&partitions, schema(), None).unwrap()
    }

    /// Over a multi-partition scan the check sits on one partition, and keeps
    /// the defaults a fetch must never be pushed past.
    #[test]
    fn a_checked_scan_is_one_partition_and_admits_no_fetch() {
        let src = source(vec![
            batch(vec![Some(1)], vec!["a"]),
            batch(vec![Some(2)], vec!["b"]),
            batch(vec![Some(3)], vec!["c"]),
        ]);
        let plan = key_checked(src, "id").unwrap();
        assert_eq!(plan.name(), "KeyCheckExec");
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        assert_eq!(plan.children()[0].name(), "CoalescePartitionsExec");
        assert!(!plan.supports_limit_pushdown());
        assert!(Arc::clone(&plan).with_fetch(Some(1)).is_none());
    }

    /// Executed over more than one partition the count would be partial, so it
    /// refuses to run at all.
    #[test]
    fn a_key_check_over_several_partitions_refuses_to_execute() {
        let src = source(vec![
            batch(vec![Some(1)], vec!["a"]),
            batch(vec![None], vec!["b"]),
        ]);
        let check = KeyCheckExec::try_new(src, "id").unwrap();
        let err = check
            .execute(0, SessionContext::new().task_ctx())
            .err()
            .expect("a multi-partition input must refuse");
        assert!(err.to_string().contains("one partition"), "{err}");
    }

    /// Null keys across every partition are one typed refusal with the exact
    /// total, raised at end of input.
    #[tokio::test]
    async fn null_keys_are_one_typed_refusal_with_the_exact_count() {
        let src = source(vec![
            batch(vec![Some(1), None], vec!["a", "b"]),
            batch(vec![None, None, Some(5)], vec!["c", "d", "e"]),
        ]);
        let plan = key_checked(src, "id").unwrap();
        let err = collect(plan.execute(0, SessionContext::new().task_ctx()).unwrap())
            .await
            .expect_err("null keys must refuse");
        let refusal = Error::found_in(&err).expect("the refusal is typed, never stringified");
        assert!(matches!(
            refusal,
            Error::InvalidKey { column, null_count: 3 } if column == "id"
        ));
    }
}
