//! `KeyCheckExec` — the null-key refusal at the input edge (K2).
//!
//! A single-partition-preserving passthrough: every batch flows through
//! unchanged while the node counts the nulls in the RAW key column, and at end
//! of input it yields exactly one `Err(External(JammiError::InvalidKey {
//! column, null_count }))` if the total is non-zero. Placed BELOW the blocking
//! `SortExec` that [`crate::operator::ordered_input`] builds, so `InferenceExec`
//! pulls zero batches before the refusal: the model is invoked zero times, the
//! source is scanned once, and the count is exact. The runner's own cast keeps
//! only a defensive check behind this.
//!
//! It never overrides `supports_limit_pushdown` (default `false`) nor
//! `with_fetch` (default `None`): a fetch must stay above it, never be pushed
//! into a node that must see every row to count.

use std::any::Any;
use std::fmt::{self, Formatter};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::Stream;
use jammi_db::error::JammiError;

/// The null-key counting passthrough. See the module doc.
#[derive(Debug)]
pub struct KeyCheckExec {
    input: Arc<dyn ExecutionPlan>,
    key_column: String,
    key_index: usize,
    properties: PlanProperties,
}

impl KeyCheckExec {
    /// Wrap `input`, checking `key_column` (which must exist in the input
    /// schema — a typed plan error otherwise).
    pub fn try_new(input: Arc<dyn ExecutionPlan>, key_column: &str) -> DfResult<Self> {
        let key_index = input.schema().index_of(key_column)?;
        let properties = PlanProperties::new(
            input.equivalence_properties().clone(),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            key_column: key_column.to_string(),
            key_index,
            properties,
        })
    }

    /// The key column this node counts nulls in.
    pub fn key_column(&self) -> &str {
        &self.key_column
    }
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
        let inner = self.input.execute(partition, context)?;
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
                        JammiError::InvalidKey {
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
